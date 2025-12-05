use crate::error::ImageError;
use crate::template::{TemplateData, TemplateDataEnum};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct MatchHit {
    /// 匹配位置的 X 坐标
    pub x: u32,
    /// 匹配位置的 Y 坐标
    pub y: u32,
    /// 归一化互相关系数，范围 [-1.0, 1.0]，越接近 1.0 匹配度越高
    pub correlation: f64,
}

impl PartialEq for MatchHit {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Eq for MatchHit {}

impl PartialOrd for MatchHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MatchHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.correlation.total_cmp(&other.correlation)
    }
}

#[derive(Debug, Clone)]
pub struct MatchResults {
    /// 模板宽度
    pub width: u32,
    /// 模板高度
    pub height: u32,
    /// 最佳匹配结果（相关系数最大）
    pub best_result: MatchHit,
    /// 所有满足阈值的匹配结果列表（不包含最佳结果）
    pub more_result: Vec<MatchHit>,
}

impl MatchResults {
    /// 创建新的匹配结果集合
    ///
    /// # 参数
    ///
    /// * `template_data` - 模板数据，用于获取尺寸信息
    /// * `all_result` - 所有匹配结果的列表
    ///
    /// # 返回值
    ///
    /// 如果 `all_result` 为空，返回 `ImageError::NoMatchingResults`。
    /// 否则返回 `MatchResults` 实例，其中 `best_result` 为 `all_result` 中的第一个元素。
    pub(crate) fn new(
        template_data: &TemplateDataEnum,
        mut all_result: Vec<MatchHit>,
    ) -> Result<Self, ImageError> {
        if all_result.is_empty() {
            return Err(ImageError::NoMatchingResults);
        }

        let best_result = all_result.remove(0);

        let (template_width, template_height) = template_data.template_dimensions();

        let result = Self {
            width: template_width,
            height: template_height,
            best_result,
            more_result: all_result,
        };

        Ok(result)
    }
}
