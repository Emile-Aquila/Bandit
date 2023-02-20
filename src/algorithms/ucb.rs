use crate::algorithms::utils::{argmax, mean, RewardHistory};

pub fn ucb_policy(rew_history: &RewardHistory, t: u32) -> u32{
    let mut scores: Vec<f64> = Vec::new();
    for rewards in &rew_history.rewards {
        let len = match rewards.len() {
            0 => 1.0_f64,
            _ => rewards.len() as f64,
        };
        let ucb_score = mean(rewards) + ((t as f64).ln() / len * 2.0_f64).sqrt();
        scores.push(ucb_score);
    }
    argmax(&scores) as u32
}