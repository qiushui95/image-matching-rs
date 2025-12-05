use crate::MatcherMode;
use crate::fft::FFTTemplateData;
use crate::segmented::SegmentedTemplateData;
use image::{ImageBuffer, Luma};

#[derive(Debug, Clone)]
pub enum TemplateData {
    FFT { data: FFTTemplateData },
    Segmented { data: SegmentedTemplateData },
}

impl TemplateData {
    pub fn new(template: &ImageBuffer<Luma<u8>, Vec<u8>>, mode: MatcherMode) -> Self {
        match mode {
            MatcherMode::FFT {
                src_width,
                src_height,
            } => TemplateData::FFT {
                data: FFTTemplateData::new(template, src_width, src_height),
            },
            MatcherMode::Segmented => TemplateData::Segmented {
                data: SegmentedTemplateData::new(template),
            },
        }
    }

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
