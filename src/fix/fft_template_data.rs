use rustfft::num_complex::Complex;

#[derive(Debug, Clone)]
pub(super) struct FFTTemplateData {
    pub template_conj_freq: Vec<Complex<f64>>,
    pub template_sum_squared_deviations: f64,
    pub template_width: u32,
    pub template_height: u32,
    pub padded_size: u32,
}
