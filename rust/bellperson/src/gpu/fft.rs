use std::cmp;
use std::ops::MulAssign;
use std::sync::{Arc, RwLock};

use ff::Field;
use log::{error, info};
use pairing::Engine;
use rust_gpu_tools::{program_closures, Device, LocalBuffer, Program};

use crate::gpu::{
    error::{GPUError, GPUResult},
    locks, program, GpuEngine,
};
use crate::multicore::THREAD_POOL;

const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
const MAX_LOG2_RADIX: u32 = 8; // Radix256
const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 7; // 128

const MAX_LOG2_RADIX_2080TI: u32 = 10; // 1024
const MAX_LOG2_LOCAL_WORK_SIZE_2080TI: u32 = 9; // 512

pub struct SingleFftKernel<E>
where
    E: Engine + GpuEngine,
{
    program: Program,
    priority: bool,
    _phantom: std::marker::PhantomData<E::Fr>,
}

impl<E: Engine + GpuEngine> SingleFftKernel<E> {
    pub fn create(device: &Device, priority: bool) -> GPUResult<Self> {
        let program = program::program::<E>(&device)?;

        Ok(SingleFftKernel {
            program,
            priority,
            _phantom: Default::default(),
        })
    }

    /// perform all gpu operation in one function
    pub fn all_in_one(
        &mut self,
        input: &mut [&mut [<E as Engine>::Fr]; 3],
        omega: &E::Fr,
        omega_a: &E::Fr,
        minvs: &[E::Fr],
        d_powers: &[E::Fr],
        d_powers_a: &[E::Fr],
        log_n: u32,
        az: &[E::Fr],
    ) -> GPUResult<()> {
        // dbg!(input[0] == input[1]);
        // dbg!(input[0] == input[2]);
        // dbg!(input[2] == input[1]);

        let n = 1 << log_n;
        let max_deg = cmp::min(MAX_LOG2_RADIX_2080TI, log_n);

        // Precalculate:
        // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
        let mut pq = vec![E::Fr::zero(); 1 << max_deg >> 1];
        let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
        pq[0] = E::Fr::one();
        if max_deg > 1 {
            pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i].mul_assign(&twiddle);
            }
        }

        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        let mut omegas = vec![E::Fr::zero(); 32];
        omegas[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow_vartime([2u64]);
        }
        let omegas = omegas;

        let closures = program_closures!(|program,
                                          input: &mut [&mut [<E as Engine>::Fr]; 3]|
         -> GPUResult<()> {
            // a,b,c same
            let mut d_powers = program.create_buffer_from_slice(&d_powers)?;

            // All usages are safe as the buffers are initialized from either the host or the GPU
            // before they are read.
            let mut src_buffer = unsafe { program.create_buffer::<E::Fr>(n)? };
            let mut dst_buffer = unsafe { program.create_buffer::<E::Fr>(n)? };
            // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`

            let mut pq_buffer = program.create_buffer_from_slice(&pq)?;

            // global
            let mut omegas_buffer = program.create_buffer_from_slice(&omegas)?;

            // minv is special
            let minv = program.create_buffer_from_slice(&[minvs[2]])?;
            program.write_from_buffer(&mut src_buffer, &*input[2])?;

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round

            while log_p < log_n {
                // remove abort check
                // if let Some(maybe_abort) = &self.maybe_abort {
                //     if maybe_abort() {
                //         return Err(EcError::Aborted);
                //     }
                // }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                // let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let local_work_size = 1 << cmp::min(deg, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let global_work_size = n >> deg;
                let kernel = program.create_kernel(
                    "all_in_one",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<E::Fr>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            let kernel_mul = program.create_kernel("mul_vec", n / 512 as usize, 512 as usize)?;
            kernel_mul.arg(&src_buffer).arg(&minv).run()?;

            let kernel_power =
                program.create_kernel("distribute_powers", n / 512 as usize, 512 as usize)?;
            kernel_power.arg(&src_buffer).arg(&d_powers).run()?;

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round

            while log_p < log_n {
                // if let Some(maybe_abort) = &self.maybe_abort {
                //     if maybe_abort() {
                //         return Err(EcError::Aborted);
                //     }
                // }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                // let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let local_work_size = 1 << cmp::min(deg, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let global_work_size = n >> deg;
                let kernel = program.create_kernel(
                    "all_in_one",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<E::Fr>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            program.read_into_buffer(&src_buffer, input[2])?;

            ////////////////////////////////  c  finished  //////////////////////////////////////

            program.write_from_buffer(&mut src_buffer, &*input[1])?;

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round

            while log_p < log_n {
                // if let Some(maybe_abort) = &self.maybe_abort {
                //     if maybe_abort() {
                //         return Err(EcError::Aborted);
                //     }
                // }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                // let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let local_work_size = 1 << cmp::min(deg, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let global_work_size = n >> deg;
                let kernel = program.create_kernel(
                    "all_in_one",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<E::Fr>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            let minv = program.create_buffer_from_slice(&[minvs[1]])?;
            let kernel_mul = program.create_kernel("mul_vec", n / 512 as usize, 512 as usize)?;
            kernel_mul.arg(&src_buffer).arg(&minv).run()?;

            let kernel_power =
                program.create_kernel("distribute_powers", n / 512 as usize, 512 as usize)?;
            kernel_power.arg(&src_buffer).arg(&d_powers).run()?;

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round

            while log_p < log_n {
                // if let Some(maybe_abort) = &self.maybe_abort {
                //     if maybe_abort() {
                //         return Err(EcError::Aborted);
                //     }
                // }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                // let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let local_work_size = 1 << cmp::min(deg, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let global_work_size = n >> deg;
                let kernel = program.create_kernel(
                    "all_in_one",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<E::Fr>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            program.read_into_buffer(&src_buffer, input[1])?;

            ////////////////////////////////  b  finished  //////////////////////////////////////

            program.write_from_buffer(&mut src_buffer, &*input[0])?;

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round

            while log_p < log_n {
                // if let Some(maybe_abort) = &self.maybe_abort {
                //     if maybe_abort() {
                //         return Err(EcError::Aborted);
                //     }
                // }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                // let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let local_work_size = 1 << cmp::min(deg, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let global_work_size = n >> deg;
                let kernel = program.create_kernel(
                    "all_in_one",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<E::Fr>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            let minv = program.create_buffer_from_slice(&[minvs[0]])?;
            let kernel_mul = program.create_kernel("mul_vec", n / 512 as usize, 512 as usize)?;
            kernel_mul.arg(&src_buffer).arg(&minv).run()?;

            let kernel_power =
                program.create_kernel("distribute_powers", n / 512 as usize, 512 as usize)?;
            kernel_power.arg(&src_buffer).arg(&d_powers).run()?;

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round

            while log_p < log_n {
                // if let Some(maybe_abort) = &self.maybe_abort {
                //     if maybe_abort() {
                //         return Err(EcError::Aborted);
                //     }
                // }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                // let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let local_work_size = 1 << cmp::min(deg, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let global_work_size = n >> deg;
                let kernel = program.create_kernel(
                    "all_in_one",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<E::Fr>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            ////////////////////////////////  a  finished  //////////////////////////////////////
            program.write_from_buffer(&mut dst_buffer, &*input[1])?;

            let kernel_mul_vec =
                program.create_kernel("mul_vector", n / 512 as usize, 512 as usize)?;
            kernel_mul_vec.arg(&src_buffer).arg(&dst_buffer).run()?;

            program.write_from_buffer(&mut dst_buffer, &*input[2])?;
            let kernel_sub_vec =
                program.create_kernel("sub_vector", n / 512 as usize, 512 as usize)?;
            kernel_sub_vec.arg(&src_buffer).arg(&dst_buffer).run()?;

            let kernel_mul = program.create_kernel("mul_vec", n / 512 as usize, 512 as usize)?;
            let minv = program.create_buffer_from_slice(&[az[0]])?;
            kernel_mul.arg(&src_buffer).arg(&minv).run()?;

            //////////////// (a*b-c)*az finished //////////////////////
            let minv = program.create_buffer_from_slice(&[minvs[0]])?;
            program.write_from_buffer(&mut d_powers, &*d_powers_a)?;
            // let d_powers = program.create_buffer_from_slice(&d_powers_a)?;

            // Precalculate:
            // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
            let mut pq = vec![E::Fr::zero(); 1 << max_deg >> 1];
            let twiddle = omega_a.pow_vartime([(n >> max_deg) as u64]);
            pq[0] = E::Fr::one();
            if max_deg > 1 {
                pq[1] = twiddle;
                for i in 2..(1 << max_deg >> 1) {
                    pq[i] = pq[i - 1];
                    pq[i].mul_assign(&twiddle);
                }
            }
            // let pq_buffer = program.create_buffer_from_slice(&pq)?;
            program.write_from_buffer(&mut pq_buffer, &*pq)?;

            // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
            let mut omegas = vec![E::Fr::zero(); 32];
            omegas[0] = *omega_a;
            for i in 1..LOG2_MAX_ELEMENTS {
                omegas[i] = omegas[i - 1].pow_vartime([2u64]);
            }
            // let omegas_buffer = program.create_buffer_from_slice(&omegas)?;
            program.write_from_buffer(&mut omegas_buffer, &*omegas)?;

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round

            while log_p < log_n {
                // if let Some(maybe_abort) = &self.maybe_abort {
                //     if maybe_abort() {
                //         return Err(EcError::Aborted);
                //     }
                // }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                // let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let local_work_size = 1 << cmp::min(deg, MAX_LOG2_LOCAL_WORK_SIZE_2080TI);
                let global_work_size = n >> deg;
                let kernel = program.create_kernel(
                    "all_in_one",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<E::Fr>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            let kernel_mul = program.create_kernel("mul_vec", n / 512 as usize, 512 as usize)?;
            kernel_mul.arg(&src_buffer).arg(&minv).run()?;

            let kernel_power =
                program.create_kernel("distribute_powers", n / 512 as usize, 512 as usize)?;
            kernel_power.arg(&src_buffer).arg(&d_powers).run()?;

            program.read_into_buffer(&src_buffer, input[0])?;

            Ok(())
        });

        self.program.run(closures, input)
    }

    /// Performs FFT on `input`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft(&mut self, input: &mut [E::Fr], omega: &E::Fr, log_n: u32) -> GPUResult<()> {
        let closures = program_closures!(|program, input: &mut [E::Fr]| -> GPUResult<()> {
            let n = 1 << log_n;
            // All usages are safe as the buffers are initialized from either the host or the GPU
            // before they are read.
            let mut src_buffer = unsafe { program.create_buffer::<E::Fr>(n)? };
            let mut dst_buffer = unsafe { program.create_buffer::<E::Fr>(n)? };
            // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
            let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

            // Precalculate:
            // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
            let mut pq = vec![E::Fr::zero(); 1 << max_deg >> 1];
            let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
            pq[0] = E::Fr::one();
            if max_deg > 1 {
                pq[1] = twiddle;
                for i in 2..(1 << max_deg >> 1) {
                    pq[i] = pq[i - 1];
                    pq[i].mul_assign(&twiddle);
                }
            }
            let pq_buffer = program.create_buffer_from_slice(&pq)?;

            // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
            let mut omegas = vec![E::Fr::zero(); 32];
            omegas[0] = *omega;
            for i in 1..LOG2_MAX_ELEMENTS {
                omegas[i] = omegas[i - 1].pow_vartime([2u64]);
            }
            let omegas_buffer = program.create_buffer_from_slice(&omegas)?;

            program.write_from_buffer(&mut src_buffer, &*input)?;
            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round
            while log_p < log_n {
                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                if locks::PriorityLock::should_break(self.priority) {
                    return Err(GPUError::GPUTaken);
                }

                let n = 1u32 << log_n;
                let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                let global_work_size = n >> deg;
                let kernel = program.create_kernel(
                    "radix_fft",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<E::Fr>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            program.read_into_buffer(&src_buffer, input)?;

            Ok(())
        });

        self.program.run(closures, input)
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct FFTKernel<E>
where
    E: Engine + GpuEngine,
{
    pub kernels: Vec<SingleFftKernel<E>>,
    _lock: locks::GPULock, // RFC 1857: struct fields are dropped in the same order as they are declared.
}

impl<E> FFTKernel<E>
where
    E: Engine + GpuEngine,
{
    pub fn create(priority: bool) -> GPUResult<FFTKernel<E>> {
        let lock = locks::GPULock::lock();

        let kernels: Vec<_> = Device::all()
            .iter()
            .filter_map(|device| {
                let kernel = SingleFftKernel::<E>::create(device, priority);
                if let Err(ref e) = kernel {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device.name(),
                        e
                    );
                }
                kernel.ok()
            })
            .collect();

        if kernels.is_empty() {
            return Err(GPUError::Simple("No working GPUs found!"));
        }
        info!("FFT: {} working device(s) selected. ", kernels.len());
        for (i, k) in kernels.iter().enumerate() {
            info!("FFT: Device {}: {}", i, k.program.device_name(),);
        }

        Ok(FFTKernel {
            kernels,
            _lock: lock,
        })
    }

    /// try all computation in gpu
    // pub fn all_in_one(
    //     &mut self,
    //     input: &mut [&mut [E::Fr]],
    //     omega: &E::Fr,
    //     omega_a: &E::Fr,
    //     minv: &[E::Fr],
    //     d_powers: &[E::Fr],
    //     d_powers_a: &[E::Fr],
    //     log_n: u32,
    //     az: &[E::Fr],
    // ) -> GPUResult<()> {
    //     info!("+++++++++++++++ using all in one kernel +++++++++++++++++");
    //     self.kernels[0].all_in_one(input, omega, omega_a, minv, d_powers, d_powers_a, log_n, az)
    // }

    /// Performs FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    ///
    /// Uses the first available GPU.
    pub fn radix_fft(&mut self, input: &mut [E::Fr], omega: &E::Fr, log_n: u32) -> GPUResult<()> {
        dbg!("num of kernels ", self.kernels.len());
        self.kernels[0].radix_fft(input, omega, log_n)
    }

    /// Performs FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    ///
    /// Uses all available GPUs to distribute the work.
    pub fn radix_fft_many(
        &mut self,
        inputs: &mut [&mut [E::Fr]],
        omegas: &[E::Fr],
        log_ns: &[u32],
    ) -> GPUResult<()> {
        let n = inputs.len();
        let num_devices = self.kernels.len();
        let chunk_size = ((n as f64) / (num_devices as f64)).ceil() as usize;

        let result = Arc::new(RwLock::new(Ok(())));

        THREAD_POOL.scoped(|s| {
            for (((inputs, omegas), log_ns), kern) in inputs
                .chunks_mut(chunk_size)
                .zip(omegas.chunks(chunk_size))
                .zip(log_ns.chunks(chunk_size))
                .zip(self.kernels.iter_mut())
            {
                let result = result.clone();
                s.execute(move || {
                    for ((input, omega), log_n) in
                        inputs.iter_mut().zip(omegas.iter()).zip(log_ns.iter())
                    {
                        if result.read().unwrap().is_err() {
                            break;
                        }

                        if let Err(err) = kern.radix_fft(input, omega, *log_n) {
                            *result.write().unwrap() = Err(err);
                            break;
                        }
                    }
                });
            }
        });

        Arc::try_unwrap(result).unwrap().into_inner().unwrap()
    }
}
