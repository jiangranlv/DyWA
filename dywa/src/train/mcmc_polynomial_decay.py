class PolynomialSchedule:
  """Polynomial learning rate schedule for Langevin sampler."""

  def __init__(self, init, final, power, num_steps):
    self._init = init
    self._final = final
    self._power = power
    self._num_steps = num_steps

  def get_rate(self, index):
    """Get learning rate for index."""
    return ((self._init - self._final) *
            ((1 - (float(index) / float(self._num_steps-1))) ** (self._power))
            ) + self._final

if __name__ == '__main__':
    sampler_stepsize_init = 0.5
    sampler_stepsize_final = 1e-5
    sampler_stepsize_power = 2.0
    num_iterations = 100
    schedule = PolynomialSchedule(sampler_stepsize_init, sampler_stepsize_final,
                                  sampler_stepsize_power, num_iterations)
    
    for i in range(10):
        print(schedule.get_rate(i))