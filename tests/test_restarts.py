import numpy as np

from facet import decode, restarts


def test_restart_evidence_shape():
    team, dead = restarts.evidence(1000, [(500, 1), (800, None)])
    m, b, a = restarts.MARGIN, restarts.BEFORE, restarts.AFTER
    assert dead[500 - m - 1] > 0 and dead[500 + m] < 0          # dead before, live after
    assert dead[500] == 0                                        # the tagged moment is left alone
    assert team[500 + m] < 0                                     # away restarts
    assert team[800 + m] == 0                                    # throw-in: no team
    assert team[:500 - m].sum() == 0


def test_restart_tips_a_close_call_but_not_clear_play():
    # tracking hints at a stoppage that is too weak on its own to pay for two state changes
    n = 600
    team, dead = np.full(n, 0.5), np.full(n, -2.0)
    dead[150:300] = 1.0
    restart = [(300 + restarts.MARGIN, 0)]
    add_team, add_dead = restarts.evidence(n, restart)
    plain = decode.viterbi(decode.emissions(team, dead))
    tagged = decode.viterbi(decode.emissions(team + add_team, dead + add_dead))
    assert (plain == decode.D).sum() == 0
    assert (tagged[200:290] == decode.D).all() and (tagged[320:] != decode.D).all()
    # where tracking clearly says play is live, one tag does not create a stoppage
    clear = np.full(n, -2.0)
    assert (decode.viterbi(decode.emissions(team + add_team, clear + add_dead)) == decode.D).sum() == 0


def test_like_smart_tagging_drops_shifts_and_hides_throw_in_teams():
    true = [(i * 1000, i % 2, "THROW IN" if i % 3 == 0 else "FREE KICK") for i in range(1, 200)]
    labels = np.array(["H"] * 200_000)
    tagged = restarts.like_smart_tagging(true, labels, seed=0)
    assert 0.6 * len(true) < len(tagged) < 0.95 * len(true)
    assert any(side is None for _, side in tagged)
