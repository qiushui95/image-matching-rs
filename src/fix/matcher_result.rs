use std::error::Error;

use crate::fix::template_data::TemplateData;
use crate::fix::matcher_single_result::MatcherSingleResult;

#[derive(Debug, Clone)]
pub struct MatcherResult {
    pub width: u32,
    pub height: u32,
    pub best_result: MatcherSingleResult,
    pub all_result: Vec<MatcherSingleResult>,
}

impl MatcherResult {
    pub(super) fn new(
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
