use crate::algorithms::utils::{argmax, max, mean, RewardHistory};
use crate::bandit::{BanditMachine, ProbabilisticBanditMachine};
use rand::Rng;

// 逐次削除方策(最適腕識別)

pub fn successive_elimination_policy(bandit: &ProbabilisticBanditMachine, rew_history: &mut RewardHistory, eps: f64,
                                     delta: f64, beta: fn(f64, f64)->f64) -> (i32, u64){
    let mut candidate_set: Vec<u32> = (0..bandit.mus.len() as u32).collect::<Vec<_>>();
    let mut trial_times: u64 = 0;
    while !candidate_set.is_empty() {
        for arm_id in &candidate_set {
            rew_history.observe(arm_id.clone(), bandit.get_reward(arm_id.clone()));
            trial_times += 1;
        }
        let ucb_scores = (0..bandit.mus.len() as u32).collect::<Vec<_>>().iter()
            .map(|&arm_id|{
                ucb_score(rew_history.means[arm_id as usize],
                          rew_history.rewards[arm_id as usize].len() as f64, delta, beta)
            })
            .collect::<Vec<_>>();
        let lcb_scores= (0..bandit.mus.len() as u32).collect::<Vec<_>>().iter()
            .map(|&arm_id|{
                lcb_score(rew_history.means[arm_id as usize],
                          rew_history.rewards[arm_id as usize].len() as f64, delta, beta)
            })
            .collect::<Vec<_>>();
        let optimal_arm = argmax(&rew_history.means) as usize;
        let mut sub_vec = ucb_scores[..optimal_arm].iter().cloned()
            .chain(ucb_scores[optimal_arm+1..].iter().cloned()).collect::<Vec<_>>();
        if lcb_scores[optimal_arm] + eps > max(&sub_vec) {
            return (optimal_arm as i32, trial_times);
        }
        candidate_set = candidate_set.iter().cloned().filter(
            |&arm_id| lcb_scores[optimal_arm] <= ucb_scores[arm_id as usize]
        ).collect::<Vec<_>>();
    }
    (-1 as i32, trial_times)
}

fn ucb_score(mean: f64, num: f64, delta: f64, beta: fn(f64, f64)->f64) -> f64{
    mean + (beta(num, delta) / 2.0_f64 / num).sqrt()
}

fn lcb_score(mean: f64, num: f64, delta: f64, beta: fn(f64, f64)->f64) -> f64{
    mean - (beta(num, delta) / 2.0_f64 / num).sqrt()
}