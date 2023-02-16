use crate::algorithms::epsilon_greedy::epsilon_greedy_policy;
use crate::algorithms::utils::{build_reward_history, RewardHistory};
use crate::bandit::BanditMachine;
mod bandit;
mod algorithms;


fn main(){
    let mus: Vec<f64> = vec![1.0, 2.0, 3.0, 1.0];
    let machine = bandit::ProbabilisticBanditMachine{
        mus,
        arm: Box::new(bandit::build_gaussian_reward(2.0)),
    };
    for i in 0..4 {
        println!("{}", machine.get_reward(i));
    }

    let mut rew_history: RewardHistory = build_reward_history(4);
    let mut rew_sum: f64 = 0.0;
    for _ in 0..100 {
        let selected_arm: u32 = epsilon_greedy_policy(&rew_history, 0.1_f64);
        let rew = machine.get_reward(selected_arm);
        println!("selected arm is {}, rew is {}", selected_arm, rew);
        rew_history.observe(selected_arm, rew);
        rew_sum += rew;
    }
    println!("total rew is {}", rew_sum);
}