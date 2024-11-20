def test_variance(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    state = scheduler.create_state()
    assert jnp.sum(jnp.abs(scheduler._get_variance(state, 0, 0) - 0.0)) < 1e-05
    assert jnp.sum(jnp.abs(scheduler._get_variance(state, 420, 400) - 0.14771)
        ) < 1e-05
    assert jnp.sum(jnp.abs(scheduler._get_variance(state, 980, 960) - 0.3246)
        ) < 1e-05
    assert jnp.sum(jnp.abs(scheduler._get_variance(state, 0, 0) - 0.0)) < 1e-05
    assert jnp.sum(jnp.abs(scheduler._get_variance(state, 487, 486) - 0.00979)
        ) < 1e-05
    assert jnp.sum(jnp.abs(scheduler._get_variance(state, 999, 998) - 0.02)
        ) < 1e-05
