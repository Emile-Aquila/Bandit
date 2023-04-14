use crate::algorithms::utils::{argmax, mean, RewardHistory};

pub fn expected_regret(rew_history: &RewardHistory, optimal_mu: f64) -> f64 {
    (0..rew_history.arm_size).into_iter().map(
        |x| (optimal_mu - rew_history.means[x as usize]) * rew_history.rewards[x as usize].len() as f64
    ).sum()
}