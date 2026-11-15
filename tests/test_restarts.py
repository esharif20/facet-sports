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
    assert any(side is None for _, side, _ in tagged)


def _pitch_with_thrower(thrower_team):
    # 4 players mid-pitch, one of each team; player 0 stands on the touchline at y = 0
    players = np.full((200, 4, 3), np.nan)
    players[:, :, 0] = 50.0
    players[:, :, 1] = [0.2, 10.0, 30.0, 40.0]
    team = np.array([thrower_team, 1 - thrower_team, 0, 1])
    return players, team


def test_throw_in_side_is_the_team_on_the_touchline():
    for thrower in (0, 1):
        players, team = _pitch_with_thrower(thrower)
        assert restarts.throw_in_side(players, team, 100, (105.0, 68.0)) == thrower


def test_throw_in_side_unknown_when_nobody_is_on_the_line():
    players, team = _pitch_with_thrower(0)
    players[:, 0, 1] = 8.0
    assert restarts.throw_in_side(players, team, 100, (105.0, 68.0)) is None
    players[:, 0, 1] = -10.0                     # a substitute warming up far behind the line
    assert restarts.throw_in_side(players, team, 100, (105.0, 68.0)) is None


def test_with_throw_in_sides_fills_only_throw_ins_without_a_team():
    players, team = _pitch_with_thrower(1)
    out = restarts.with_throw_in_sides(
        [(100, None, "THROW-IN"), (100, 0, "THROW-IN"), (100, None, "CORNER KICK")],
        players, team, (105.0, 68.0))
    assert [side for _, side, _ in out] == [1, 0, None]


def _tag(code, start, end, kind=None):
    return {"code": code, "start_frame": start, "end_frame": end,
            "tags": {"Type": kind} if kind else {}}


def test_goals_pair_each_kickoff_with_its_goal():
    tags = [_tag("GOALS", 1000, 1200), _tag("SET PIECES", 2700, 2800, "Kick Off (After Goal)"),
            _tag("SET PIECES", 50000, 50100, "Kick Off (After Goal)"),     # no goal tag near
            _tag("SET PIECES", 3000, 3100, "Corner Kick")]
    assert restarts.goals_from_tags(tags) == [(1200, 2700), (None, 50000)]


def test_goal_stoppage_covers_the_celebration_but_not_the_goal_or_kickoff():
    mask = restarts.goal_stoppages(5000, [(1200, 2700), (None, 4500)])
    assert mask[1200 + restarts.AFTER_GOAL:2700 - restarts.BEFORE_KICKOFF].all()
    assert not mask[:1200 + restarts.AFTER_GOAL].any()
    assert not mask[2700 - restarts.BEFORE_KICKOFF:4500 - restarts.NO_GOAL_TAG].any()
    assert mask[4500 - restarts.NO_GOAL_TAG:4500 - restarts.BEFORE_KICKOFF].all()
