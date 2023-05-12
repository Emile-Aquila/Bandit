use crate::algorithms::utils;
use rand::distributions::WeightedIndex;
use rand::prelude::{Distribution, ThreadRng};
use rand::thread_rng;

pub struct Exp3Agent{
    arm_size: u32,
    gamma: f64,
    eta: f64,
    weights: Vec<f64>,
    probs: Vec<f64>,
    rng: ThreadRng
}

impl Exp3Agent {
    pub fn select_arm(&mut self) -> u32 {
        let w_sum: f64 = self.weights.iter().sum();  // sum of weights
        self.probs = self.weights.clone().iter().map(
            |&x| (1.0 - self.gamma) * x / w_sum + self.gamma / (self.arm_size as f64)
        ).collect::<Vec<_>>();  // update probabilities
        self.weights = self.weights.clone().iter().map(|&w| w/w_sum).collect::<Vec<_>>();
        let dist = WeightedIndex::new(&self.probs).unwrap();
        dist.sample(&mut self.rng) as u32
    }

    pub fn observe(&mut self, selected_arm: u32, reward: f64){
        self.weights[selected_arm as usize] *= (reward / self.probs[selected_arm as usize]).exp();
    }
}

pub fn build_exp3agent(arm_size: u32, gamma: f64, eta: f64) -> Exp3Agent {
    Exp3Agent{
        arm_size,
        gamma,
        eta,
        weights: vec![1.0/arm_size as f64; arm_size as usize],
        probs: vec![0.0; arm_size as usize],
        rng: thread_rng()
    }
}


pub struct Exp3PAgent{
    arm_size: u32,
    beta: f64,
    gamma: f64,
    eta: f64,
    weights: Vec<f64>,
    probs: Vec<f64>,
    rng: ThreadRng
}


impl Exp3PAgent {
    pub fn select_arm(&mut self) -> u32 {
        let sum: f64 = self.weights.iter().sum();
        self.probs = self.weights.clone().iter().map(
            |&w| (1.0 - self.gamma)*w/sum + self.gamma/(self.arm_size as f64)
        ).collect::<Vec<_>>();
        self.weights = self.weights.clone().iter().map(|&w| w/sum).collect::<Vec<_>>();
        let dist = WeightedIndex::new(&self.probs).unwrap();
        dist.sample(&mut self.rng) as u32
    }

    pub fn observe(&mut self, selected_arm: u32, reward: f64){
        let reward_est = reward/self.probs[selected_arm as usize];
        let upper_rewards = self.probs.iter().enumerate().map(
            |(arm, prob)| (if arm == selected_arm as usize {reward_est} else {0.0}) + self.beta / prob
        ).collect::<Vec<_>>();
        self.weights = self.weights.clone().iter().zip(upper_rewards.iter()).map(
            |(w, r)| w * ((self.eta * r).exp())
        ).collect::<Vec<_>>();
    }
}

pub fn build_exp3p_agent(arm_size: u32, beta: f64, gamma: f64, eta: f64) -> Exp3PAgent {
    Exp3PAgent{
        arm_size,
        beta,
        gamma,
        eta,
        weights: vec![1.0/arm_size as f64; arm_size as usize],
        probs: vec![0.0; arm_size as usize],
        rng: thread_rng()
    }
}