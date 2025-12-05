#[derive(Debug, Clone)]
pub(super) struct SegmentedTemplateData {
    pub template_segments_fast: Vec<(u32, u32, u32, u32, f64)>,
    pub template_segments_slow: Vec<(u32, u32, u32, u32, f64)>,
    pub template_width: u32,
    pub template_height: u32,
    pub segment_sum_squared_deviations_fast: f64,
    pub segment_sum_squared_deviations_slow: f64,
    pub expected_corr_fast: f64,
    pub expected_corr_slow: f64,
    pub segments_mean_fast: f64,
    pub segments_mean_slow: f64,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum SegmentType {
    Fast,
    Slow,
}
