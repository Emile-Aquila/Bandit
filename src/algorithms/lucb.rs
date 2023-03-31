use crate::algorithms::utils::{argmax, max, mean, RewardHistory};
use crate::bandit::{BanditMachine, ProbabilisticBanditMachine};
use rand::Rng;

// 逐次削除方策(最適腕識別)

pub fn lucb_policy(bandit: &ProbabilisticBanditMachine, rew_history: &mut RewardHistory, eps: f64,
                                     delta: f64, beta: fn(f64, f64) -> f64) -> (i32, u64) {
    let mut trial_times: u64 = 0;
    for arm_id in 0..bandit.arm_size() {
        rew_history.observe(arm_id, bandit.get_reward(arm_id));
        trial_times += 1;
    }  // 各アーム一度ずつ引く
    loop {
        let ucb_scores = (0..bandit.arm_size()).collect::<Vec<_>>().iter()
            .map(|&arm_id| {
                ucb_score(rew_history.means[arm_id as usize],
                          rew_history.rewards[arm_id as usize].len() as f64, delta, beta)
            }).collect::<Vec<_>>();
        let lcb_scores = (0..bandit.arm_size()).collect::<Vec<_>>().iter()
            .map(|&arm_id|{
                lcb_score(rew_history.means[arm_id as usize],
                          rew_history.rewards[arm_id as usize].len() as f64, delta, beta)
            }).collect::<Vec<_>>();
        let optimal_arm = argmax(&rew_history.means) as usize;
        let sub_vec =  ucb_scores[..optimal_arm].iter().cloned().chain(ucb_scores[optimal_arm+1..].iter().cloned()).collect::<Vec<_>>();
        let mut optimal_arm2 = argmax(&sub_vec) as usize;
        if optimal_arm2 >= optimal_arm { optimal_arm2 += 1; }  // indexの調整
        if ucb_scores[optimal_arm2] <= lcb_scores[optimal_arm] + eps {
            return (optimal_arm as i32, trial_times);
        }
        rew_history.observe(optimal_arm as u32, bandit.get_reward(optimal_arm as u32));
        rew_history.observe(optimal_arm2 as u32, bandit.get_reward(optimal_arm2 as u32));
        trial_times += 2;
    }
}

fn ucb_score(mean: f64, num: f64, delta: f64, beta: fn(f64, f64) -> f64) -> f64 {
    mean + (beta(num, delta) / 2.0_f64 / num).sqrt()
}

fn lcb_score(mean: f64, num: f64, delta: f64, beta: fn(f64, f64) -> f64) -> f64 {
    mean - (beta(num, delta) / 2.0_f64 / num).sqrt()
}