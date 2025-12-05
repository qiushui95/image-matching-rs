mod fft;
mod integral;
mod segmented;

pub(crate) use fft::FFTTemplateData;
pub(crate) use segmented::SegmentedTemplateData;

use crate::MatcherMode;
use crate::error::ImageError;
use crate::result::MatchHit;
use image::GrayImage;

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
/// 模板数据枚举
///
/// 封装不同模式下的模板数据（FFT 或 Segmented）。
pub(crate) enum TemplateDataEnum {
    /// FFT 模式数据
    FFT { data: FFTTemplateData },
    /// 分段模式数据
    Segmented { data: SegmentedTemplateData },
}

impl TemplateDataEnum {
    /// 创建新的模板数据
    ///
    /// 根据指定的匹配模式创建对应的模板数据。
    pub fn new(template: GrayImage, mode: MatcherMode) -> Self {
        match mode {
            MatcherMode::FFT {
                src_width,
                src_height,
            } => Self::FFT {
                data: FFTTemplateData::new(template, src_width, src_height),
            },
            MatcherMode::Segmented => Self::Segmented {
                data: SegmentedTemplateData::new(template),
            },
        }
    }
}

/// 生成模板在图像上可滑动的所有左上角坐标列表
///
/// 根据图像尺寸与模板尺寸，返回合法的 `(x, y)` 坐标集合。
pub(crate) fn generate_search_coordinates(
    image_width: u32,
    image_height: u32,
    template_width: u32,
    template_height: u32,
) -> Vec<(u32, u32)> {
    (0..=(image_height - template_height))
        .flat_map(|y| (0..=(image_width - template_width)).map(move |x| (x, y)))
        .collect()
}

/// 模板数据特征
///
/// 定义了所有模板数据类型必须实现的通用接口。
pub(crate) trait TemplateData {
    /// 获取模板尺寸 (width, height)
    fn template_dimensions(&self) -> (u32, u32);

    fn common_check_img(&self, img: &GrayImage) -> Result<(), ImageError> {
        let (img_width, img_height) = img.dimensions();
        let (template_width, template_height) = self.template_dimensions();
        if img_width < template_width {
            return Err(ImageError::ImageWidthTooSmall(img_width, template_width));
        }
        if img_height < template_height {
            return Err(ImageError::ImageHeightTooSmall(img_height, template_height));
        }
        Ok(())
    }

    /// 检查图像尺寸是否满足匹配要求
    fn check_img(&self, img: &GrayImage) -> Result<(), ImageError> {
        self.common_check_img(img)
    }

    /// 执行匹配操作
    fn perform_matching(&self, img: GrayImage, threshold: f64)
    -> Result<Vec<MatchHit>, ImageError>;
}

impl TemplateData for TemplateDataEnum {
    fn template_dimensions(&self) -> (u32, u32) {
        match self {
            TemplateDataEnum::FFT { data } => data.template_dimensions(),
            TemplateDataEnum::Segmented { data } => data.template_dimensions(),
        }
    }

    fn check_img(&self, img: &GrayImage) -> Result<(), ImageError> {
        match self {
            TemplateDataEnum::FFT { data } => data.check_img(img),
            TemplateDataEnum::Segmented { data } => data.check_img(img),
        }
    }

    fn perform_matching(
        &self,
        img: GrayImage,
        threshold: f64,
    ) -> Result<Vec<MatchHit>, ImageError> {
        match self {
            TemplateDataEnum::FFT { data } => data.perform_matching(img, threshold),
            TemplateDataEnum::Segmented { data } => data.perform_matching(img, threshold),
        }
    }
}
