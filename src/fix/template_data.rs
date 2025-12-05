use super::fft_template_data::FFTTemplateData;
use super::segmented_template_data::SegmentedTemplateData;

#[derive(Debug, Clone)]
pub(super) enum TemplateData {
    FFT { data: FFTTemplateData },
    Segmented { data: SegmentedTemplateData },
}

impl TemplateData {
    pub fn get_template_width(&self) -> u32 {
        match self {
            TemplateData::FFT { data } => data.template_width,
            TemplateData::Segmented { data } => data.template_width,
        }
    }

    pub fn get_template_height(&self) -> u32 {
        match self {
            TemplateData::FFT { data } => data.template_height,
            TemplateData::Segmented { data } => data.template_height,
        }
    }
}
