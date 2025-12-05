pub struct MatcherResultFilter {
    x_delta: u32,
    y_delta: u32,
}

impl MatcherResultFilter {
    pub fn new(x_delta: u32, y_delta: u32) -> Self {
        Self { x_delta, y_delta }
    }

    pub fn default() -> Self {
        Self::new(5, 5)
    }

    pub fn need_filter(&self, item: &MatcherSingleResult, exist: &MatcherSingleResult) -> bool {
        if item.x < exist.x - self.x_delta {
            return false;
        }
        if item.x > exist.x + self.x_delta {
            return false;
        }
        if item.y < exist.y - self.y_delta {
            return false;
        }
        if item.y > exist.y + self.y_delta {
            return false;
        }
        true
    }
}
