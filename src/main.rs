use crate::algorithms::epsilon_greedy::epsilon_greedy_policy;
use crate::algorithms::thompson_sampling::ts_policy;
use crate::algorithms::ucb::ucb_policy;
use crate::algorithms::utils::{build_reward_history, plot_data, RewardHistory};
use crate::bandit::BanditMachine;
mod bandit;
mod algorithms;


fn main(){
    // params
    let T: u32 = 100;
    let rew_sigma: f64 = 0.5;

    // codes
    let mus: Vec<f64> = vec![0.5, 1.0, 2.0, 3.0];
    let machine = bandit::ProbabilisticBanditMachine{
        mus,
        arm: Box::new(bandit::build_gaussian_reward(rew_sigma)),
    };

    let mut rew_history: RewardHistory = build_reward_history(4);
    let mut miss_probs: Vec<f64> = vec![0.0_f64;0];
    let mut rew_sum: f64 = 0.0;

    for t in 0..T {
        // let selected_arm: u32 = epsilon_greedy_policy(&rew_history, 0.1_f64);
        // let selected_arm: u32 = ucb_policy(&rew_history, t);
        let selected_arm: u32 = ts_policy(&rew_history, rew_sigma, 0.0, 2.0);
        let rew = machine.get_reward(selected_arm);
        rew_history.observe(selected_arm, rew);
        let miss_prob = ((t+1-(rew_history.rewards[3].len() as u32)) as f64)/ ((t+1) as f64);

        println!("selected arm is {}, rew is {}, miss_probs {}", selected_arm, rew, miss_prob);
        miss_probs.push(miss_prob);
        rew_sum += rew;
    }
    println!("total rew is {}, miss prob is {}", rew_sum, miss_probs.last().unwrap());
    let x_data = &(0..T).map(|x| x as f64).collect::<Vec<f64>>();
    // plot_data(&x_data, &miss_probs, "test.png", "miss probs");
}