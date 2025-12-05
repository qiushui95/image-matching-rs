use crate::result::MatchHit;

/// 匹配结果过滤器
///
/// 用于根据空间距离过滤重叠或邻近的匹配结果，保留局部最优解。
pub struct MatchHitFilter {
    x_delta: u32,
    y_delta: u32,
}

impl MatchHitFilter {
    /// 创建新的过滤器
    ///
    /// # 参数
    ///
    /// * `x_delta` - X 轴方向的最小间距
    /// * `y_delta` - Y 轴方向的最小间距
    pub fn new(x_delta: u32, y_delta: u32) -> Self {
        Self { x_delta, y_delta }
    }

    /// 创建默认过滤器
    ///
    /// 默认间距为 5 像素。
    pub fn new_default() -> Self {
        Self::new(5, 5)
    }

    /// 检查是否需要过滤当前结果
    ///
    /// # 参数
    ///
    /// * `item` - 当前待检查的匹配结果
    /// * `exist` - 已存在的保留结果
    ///
    /// # 返回值
    ///
    /// 如果 `item` 在 `exist` 的 `(x_delta, y_delta)` 邻域内，则返回 `true`（表示需要过滤）。
    pub fn need_filter(&self, item: &MatchHit, exist: &MatchHit) -> bool {

        if item.x < exist.x.saturating_sub(self.x_delta) {
            return false;
        }
        if item.x > exist.x.saturating_add(self.x_delta) {
            return false;
        }
        if item.y < exist.y.saturating_sub(self.y_delta) {
            return false;
        }
        if item.y > exist.y.saturating_add(self.y_delta) {
            return false;
        }
        true
    }

    /// 过滤匹配结果列表
    ///
    /// 对输入的匹配结果列表进行过滤，去除重叠或过近的结果。
    /// 输入列表应当已经按置信度排序。
    ///
    /// # 参数
    ///
    /// * `list` - 待过滤的匹配结果列表
    ///
    /// # 返回值
    ///
    /// 过滤后的匹配结果列表
    pub fn filter_results(&self, list: Vec<MatchHit>) -> Vec<MatchHit> {
        let mut results = vec![];
        for item in list {
            let mut need_filter = false;
            for exist in results.iter() {
                if self.need_filter(&item, exist) {
                    need_filter = true;
                    break;
                }
            }
            if need_filter {
                continue;
            }
            results.push(item);
        }
        results
    }
}
