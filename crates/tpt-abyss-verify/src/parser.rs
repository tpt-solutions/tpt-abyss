use tpt_abyss_types::{ReasoningStep, StepKind};

/// Errors raised while parsing a free-text chain-of-thought.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("unexpected token at line {line}: {text}")]
    Unexpected { line: usize, text: String },
}

/// Parse free-text chain-of-thought into a [`ReasoningTrace`].
///
/// The parser is intentionally tolerant: it splits on newlines, treats lines
/// containing `=` with numeric operands as [`StepKind::Computation`], lines
/// beginning with "therefore"/"so"/"thus"/"conclusion" as
/// [`StepKind::Conclusion`], "because"/"since"/"if" as [`StepKind::Deduction`],
/// and "goal"/"problem"/"we need" as [`StepKind::Goal`]; everything else is a
/// [`StepKind::Statement`]. Computation lines are further decomposed into
/// `lhs`, `op`, `rhs`, `result` so the arithmetic checker can validate them.
pub struct TraceParser;

impl Default for TraceParser {
    fn default() -> Self {
        Self
    }
}

impl TraceParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse `text` into a trace for the given task id/type.
    pub fn parse(
        &self,
        task_id: &str,
        task_type: &str,
        text: &str,
    ) -> Result<tpt_abyss_types::ReasoningTrace, ParseError> {
        let mut trace = tpt_abyss_types::ReasoningTrace::new(task_id, task_type);
        let mut idx = 0;
        for (line_no, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(answer) = line
                .strip_prefix("answer:")
                .or_else(|| line.strip_prefix("Answer:"))
            {
                trace.final_answer = Some(answer.trim().to_string());
                continue;
            }
            let step = self.line_to_step(idx, line, line_no)?;
            trace.push_step(step);
            idx += 1;
        }
        Ok(trace)
    }
}

impl TraceParser {
    fn line_to_step(
        &self,
        index: usize,
        line: &str,
        _line_no: usize,
    ) -> Result<ReasoningStep, ParseError> {
        let kind = classify(line);
        let (lhs, op, rhs, result) = if kind == StepKind::Computation {
            parse_computation(line)
        } else {
            (None, None, None, None)
        };
        Ok(ReasoningStep {
            index,
            kind,
            text: line.to_string(),
            lhs,
            op,
            rhs,
            result,
        })
    }
}

fn classify(line: &str) -> StepKind {
    let l = line.to_ascii_lowercase();
    if l.starts_with("goal") || l.starts_with("problem") || l.starts_with("we need") {
        StepKind::Goal
    } else if l.starts_with("therefore")
        || l.starts_with("so,")
        || l.starts_with("thus")
        || l.starts_with("conclusion")
    {
        StepKind::Conclusion
    } else if l.starts_with("because")
        || l.starts_with("since")
        || l.starts_with("if ")
        || l.contains(" implies ")
    {
        StepKind::Deduction
    } else if line.contains('=') && looks_arithmetic(line) {
        StepKind::Computation
    } else {
        StepKind::Statement
    }
}

/// Heuristic: a computation line looks like "<num> <op> <num> = <num>".
fn looks_arithmetic(line: &str) -> bool {
    let mut digits = 0usize;
    for c in line.chars() {
        if c.is_ascii_digit() {
            digits += 1;
        }
    }
    digits >= 2 && line.contains('=')
}

/// Parse `lhs op rhs = result` from a computation line.
/// Supports operators `+ - * / x ÷` and handles a leading result form
/// (`result = lhs op rhs`) too.
fn parse_computation(line: &str) -> (Option<f64>, Option<String>, Option<f64>, Option<f64>) {
    // Normalize multiplication symbols.
    let norm = line.replace('x', "*").replace('÷', "/").replace('×', "*");
    let sides: Vec<&str> = norm.split('=').collect();
    if sides.len() < 2 {
        return (None, None, None, None);
    }
    // Determine which side is the result.
    let (left, right) = if sides[0]
        .trim()
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        (sides[0], sides[1])
    } else {
        (sides[1], sides[0])
    };
    let op_chars = ['+', '-', '*', '/'];
    let op_pos = left.find(|c| op_chars.contains(&c));
    let op = op_pos.map(|i| left[i..=i].to_string());
    if let (Some(pos), Some(op)) = (op_pos, &op) {
        let a = left[..pos].trim().parse::<f64>().ok();
        let b = left[pos + 1..].trim().parse::<f64>().ok();
        let r = right.trim().parse::<f64>().ok();
        (a, Some(op.clone()), b, r)
    } else {
        (None, None, None, None)
    }
}

/// Convenience free function mirroring [`TraceParser::parse`].
pub fn parse_trace(
    task_id: &str,
    task_type: &str,
    text: &str,
) -> Result<tpt_abyss_types::ReasoningTrace, ParseError> {
    TraceParser::new().parse(task_id, task_type, text)
}
