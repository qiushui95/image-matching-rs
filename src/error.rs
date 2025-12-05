use thiserror::Error;

#[derive(Error, Debug)]
/// 图像匹配错误类型
///
/// 定义库在执行匹配过程中可能返回的错误。
pub enum ImageError {
    /// 输入图像尺寸与 FFT 预设的源图尺寸不一致
    #[error("图像尺寸 {0}x{1} 与预设尺寸 {2}x{3} 不匹配")]
    DimensionsMismatch(u32, u32, u32, u32),
    /// 源图宽度小于模板宽度
    #[error("图像宽度 {0} 小于模板宽度 {1}，无法进行匹配")]
    ImageWidthTooSmall(u32, u32),
    /// 源图高度小于模板高度
    #[error("图像高度 {0} 小于模板高度 {1}，无法进行匹配")]
    ImageHeightTooSmall(u32, u32),
    /// 模板宽度大于图像宽度
    #[error("模板宽度大于图像宽度")]
    TemplateWidthTooLarge,
    /// 模板高度大于图像高度
    #[error("模板高度大于图像高度")]
    TemplateHeightTooLarge,
    /// 在给定阈值下未找到任何匹配
    #[error("未找到匹配结果")]
    NoMatchingResults,
}
