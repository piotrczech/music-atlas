# Access

**TL;DR**
- Login host `ui.wcss.pl` (ACC-05); AlmaLinux 9, 15 shared cores (ACC-01), **1 GiB
  of RAM per user** enforced by a cgroup (ACC-03). Compute in `sub-interactive`.
- Three failed password logins lock the account 24 h (ACC-02). Use the config
  below: a broken key then fails instantly instead of asking.
- Never loop connections or `ssh-keyscan`: it reads as a scan and gets the IP
  blocked at TCP level. Check the host key once from `known_hosts`.
- `ssh wcss 'cmd'` keeps `$PDDIR`, `$TMPDIR` and the WCSS shell functions on a
  stock account (ACC-08); only the `module` alias is missing.
- Browser route: <https://ood.e-science.pl>, SSO login, no key needed (ACC-06).

Official: [dostęp](https://man.e-science.pl/pl/kdm/dostep), [ssh](https://man.e-science.pl/pl/kdm/ssh).

## SSH configuration

```sshconfig
Host wcss
    HostName ui.wcss.pl
    User YOUR_USERNAME
    IdentityFile ~/.ssh/YOUR_KEY
    IdentitiesOnly yes
    BatchMode yes
    NumberOfPasswordPrompts 0
    ServerAliveInterval 60
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

`BatchMode` and `NumberOfPasswordPrompts 0` are the guardrail. `ControlMaster`
matters more than usual: without it every remote command opens a new session
on a login node shared by about 50 people. Keys are managed at
<https://users.e-science.pl>; registration and the password at
<https://e-science.pl>. The job-statistics portal `hpc-info.kdm.wcss.pl` needs
the KDM VPN (ACC-04). Helpdesk: helpdesk@e-science.pl.

## Command shells keep the environment

`ssh host 'cmd'` runs a non-login bash, but bash started by sshd reads
`~/.bashrc`, and the stock file sources `/etc/bashrc`, which runs
`/etc/profile.d`. So `$PDDIR`, `$PDDIRS` and `set-default-service` (a shell
function, not a binary, ACC-07) are there (ACC-08). `module` is not: on `ui` it
is only an alias that says to use `sub-interactive`, and aliases do not expand
in non-interactive shells. If you replaced `~/.bashrc` (`verify.sh` notes it),
use `ssh wcss 'bash -lc "..."'`.

## Open OnDemand

<https://ood.e-science.pl> authenticates through the e-science SSO, not your
key (OOD-01), so it works from any machine and survives an IP block. A session
is an ordinary SLURM job billed to the service you pick, for as long as it is
open (OOD-02). Its scripts and logs live under `~/ondemand/data/sys/dashboard/batch_connect/`
(OOD-03), inside `$HOME`'s quota; `output/<uuid>/output.log` says why a session
failed. NoMachine and port forwarding are documented in the manual.

## When you are locked out

- `ssh` times out and `ping` fails, other networks work: IP block. Use another
  network, email the helpdesk with IP and time, do not retry in a loop.
- `Permission denied (publickey)`: key problem. Check users.e-science.pl. Do
  not remove `BatchMode` to "just try the password".
- Password rejected three times: 24 h lock; email the helpdesk.
