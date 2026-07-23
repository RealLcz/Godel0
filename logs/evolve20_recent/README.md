# Recent evolve-20 runs (Jul 2026)

## Job 211215 — grounded existing-test grounding (still require_generated_contracts=true)
- Slurm: `godel0-g10`
- Config: `godel0_evolve20_pass1/configs/evolve20_ansible_pass1_grounded10.yaml`
- Run dir (cluster): `runs_ansible_evolve20_grounded10/ansible_evolve20_grounded10_211215`
- Logs in this repo: `logs/godel0_evolve20_211215.{out,err,log}`
- Outcome: Root bootstrap failed; Node proposer `exit=-15` (timeout); 0 accepted tasks; dominated by `clean_contract_failure`.

## Job 211424 — existing passing tests as contract oracle
- Slurm: `godel0-e10`
- Config: `configs/evolve20_ansible_pass1_existing10.yaml` (`require_generated_contracts: false`)
- Run dir (cluster): `runs_ansible_evolve20_existing10/ansible_evolve20_existing10_211424`
- Logs in this repo: `logs/godel0_evolve20_211424.{out,err,log}`
- Outcome: Root bootstrap failed; Node proposer `exit=-15` (timeout ~2.5h); 33 plans selected existing tests, 9 candidate artifacts written, **0** passed causal ablation / trusted validation; 0 accepted tasks.
