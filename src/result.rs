use crate::template_data::TemplateData;
use std::cmp::Ordering;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct MatcherSingleResult {
    pub x: u32,
    pub y: u32,
    pub correlation: f64,
}

impl PartialEq for MatcherSingleResult {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Eq for MatcherSingleResult {}

impl PartialOrd for MatcherSingleResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.correlation.total_cmp(&other.correlation))
    }
}

impl Ord for MatcherSingleResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.correlation.total_cmp(&other.correlation)
    }
}

#[derive(Debug, Clone)]
pub struct MatcherResult {
    pub width: u32,
    pub height: u32,
    pub best_result: MatcherSingleResult,
    pub all_result: Vec<MatcherSingleResult>,
}

impl MatcherResult {
    pub fn new(
        template_data: &TemplateData,
        mut all_result: Vec<MatcherSingleResult>,
    ) -> Result<Self, Box<dyn Error>> {
        if all_result.is_empty() {
            return Err("No matching results found".into());
        }

        let best_result = all_result.remove(0);

        Ok(Self {
            width: template_data.get_template_width(),
            height: template_data.get_template_height(),
            best_result,
            all_result,
        })
    }
}
