use rand_distr::Distribution;
use crate::algorithms::utils::{argmax, argmin, normalize, softmax};

pub trait BanditMachine{
    fn get_reward(&self, arm_index: u32) -> f64;  // indexに対応するアームを引く
    fn arm_size(&self) -> u32;  // armの数を返す
}

pub trait ProbabilisticBanditArm {
    fn get_reward(&self, mu: f64) -> f64;
}


pub struct ProbabilisticBanditMachine {
    pub(crate) mus: Vec<f64>,
    pub(crate) arm: Box<dyn ProbabilisticBanditArm>,
}

impl BanditMachine for ProbabilisticBanditMachine{
    fn get_reward(&self, arm_index: u32) -> f64 {
        self.arm.get_reward(self.mus[arm_index as usize])
    }

    fn arm_size(&self) -> u32 {
        self.mus.len() as u32
    }
}

pub struct GaussianReward{
    dist_normal: rand_distr::Normal<f64>,
}

impl ProbabilisticBanditArm for GaussianReward{
    fn get_reward(&self, mu: f64) -> f64 {
        self.dist_normal.sample(&mut rand::thread_rng()) + mu
    }
}

pub fn build_gaussian_reward(sigma: f64) -> GaussianReward{
    GaussianReward {
        dist_normal: rand_distr::Normal::new(0.0, sigma).unwrap(),
    }
}


pub struct AdversarialBanditMachine {
    arm_weights: Vec<f64>,  //
    gamma: f64,  // 減衰係数

}

pub fn build_adversarial_bandit_machine(arm_size: usize, gamma: f64) -> AdversarialBanditMachine {
    AdversarialBanditMachine{
        arm_weights: vec![0.0; arm_size],
        gamma
    }
}

impl AdversarialBanditMachine{
    pub fn best_arm(&self) -> u32 {
        argmin(&self.arm_weights) as u32
    }

    pub fn get_rewards(&self) -> Vec<f64> {
        softmax(&(self.arm_weights.clone().iter().map(|&x| -x).collect::<Vec<_>>()))
    }

    pub fn observe(&mut self, selected_arm: u32){
        self.arm_weights = self.arm_weights.iter().map(
            |&w| self.gamma * w
        ).collect::<Vec<_>>();
        self.arm_weights[selected_arm as usize] += 1.0;
        self.arm_weights = normalize(&self.arm_weights);
    }
}

impl BanditMachine for AdversarialBanditMachine {
    fn get_reward(&self, arm_index: u32) -> f64 {
        self.get_rewards()[arm_index as usize]
    }

    fn arm_size(&self) -> u32 {
        self.arm_weights.len() as u32
    }
}