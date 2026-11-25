import math
import numpy as np
import pytest
from aethernet import Codec, PeerTable, ReplayWindow, NavSnapshot, AnchorInfo, PACKET_SIZE
from aethernet.protocol import sigma_to_code, code_to_sigma, FLAG_SATURATED

KEY = b"swarm-key-0123456789"


def snap(pos=(10.0, -4.0), P=None, alt=6.0, vel=(1.0, 2.0), node=3, anchor=False):
    P = np.array([[0.04, 0.01], [0.01, 0.09]]) if P is None else P
    return NavSnapshot(node, np.array(pos), np.array(vel), P, alt, 0.5, 0.3, anchor, True, False)


def test_size_is_32():
    assert PACKET_SIZE == 32 and len(Codec(KEY).encode(snap(), 1)) == 32


def test_roundtrip_values():
    c = Codec(KEY)
    p = c.decode(c.encode(snap(), 9))
    assert p.node_id == 3 and p.seq == 9
    assert np.allclose(p.pos, [10.0, -4.0], atol=1e-3) and np.allclose(p.vel, [1, 2], atol=0.01)
    assert abs(p.alt - 6.0) < 0.01


def test_covariance_never_overconfident():
    c = Codec(KEY)
    rng = np.random.default_rng(0)
    for _ in range(500):
        sx, sy = 10 ** rng.uniform(-2, 2.5, 2)
        rho = rng.uniform(-0.95, 0.95)
        P = np.array([[sx**2, rho*sx*sy], [rho*sx*sy, sy**2]])
        d = c.decode(c.encode(snap(P=P), 1))
        assert d.P_pos[0, 0] >= P[0, 0] and d.P_pos[1, 1] >= P[1, 1]
        assert np.all(np.linalg.eigvalsh(d.P_pos) > 0)


def test_large_covariance_not_clipped_like_v1():
    c = Codec(KEY)
    d = c.decode(c.encode(snap(P=np.eye(2) * 25.0), 1))      # sigma = 5 m (v1 clipped this to 1.8 m)
    assert math.sqrt(d.P_pos[0, 0]) >= 5.0


def test_sigma_code_monotone_and_ceil():
    prev = 0
    for s in np.logspace(-2.9, 2.9, 300):
        c, sat = sigma_to_code(s)
        assert not sat and c >= prev and code_to_sigma(c) >= s * (1 - 1e-12)
        prev = c


def test_saturation_flagged_and_rejected_by_table():
    c = Codec(KEY)
    raw = c.encode(snap(P=np.eye(2) * 1e8), 1)
    assert c.decode(raw).flags & FLAG_SATURATED
    t = PeerTable(c)
    assert t.ingest(raw, 0.0) is None and t.rejected["saturated"] == 1


def test_tamper_any_bit_rejected():
    c = Codec(KEY)
    raw = bytearray(c.encode(snap(), 1))
    for i in range(len(raw) * 8):
        b = bytearray(raw); b[i // 8] ^= 1 << (i % 8)
        assert c.decode(bytes(b)) is None


def test_wrong_key_and_truncation():
    raw = Codec(KEY).encode(snap(), 1)
    assert Codec(b"another-key-0123456789").decode(raw) is None
    assert Codec(KEY).decode(raw[:-1]) is None and Codec(KEY).decode(raw + b"\0") is None


def test_forged_valid_crc_style_packet_fails():
    # attacker without the key builds a perfectly formatted packet
    forged = Codec(None, insecure=True).encode(snap(pos=(500.0, 500.0)), 1)
    assert Codec(KEY).decode(forged) is None


def test_key_required():
    with pytest.raises(ValueError):
        Codec(None)
    with pytest.raises(ValueError):
        Codec(b"short")


def test_replay_and_wrap():
    w = ReplayWindow()
    assert w.accept(1, 10, 0.0) and not w.accept(1, 10, 0.1) and not w.accept(1, 9, 0.2)
    assert w.accept(1, 11, 0.3)
    w2 = ReplayWindow()
    assert w2.accept(2, 65535, 0.0) and w2.accept(2, 0, 0.1) and w2.accept(2, 1, 0.2)   # wrap
    assert not w2.accept(2, 65535, 0.3)


def test_replay_reboot_timeout():
    w = ReplayWindow(reset_after_s=5.0)
    assert w.accept(1, 100, 0.0)
    assert not w.accept(1, 5, 1.0)          # looks like replay
    assert w.accept(1, 5, 10.0)             # sender silent > 5 s: treat as reboot


def test_peertable_end_to_end_and_age():
    c = Codec(KEY)
    t = PeerTable(c)
    raw = c.encode(snap(node=4), 1)
    assert t.ingest(raw, 100.0) is not None
    assert t.ingest(raw, 100.01) is None and t.rejected["replay"] == 1
    o = t.observe(4, 7.5, 100.08)
    assert o is not None and abs(o.age_s - 0.08) < 1e-9 and not o.is_anchor
    assert t.observe(99, 7.5, 100.0) is None


def test_anchor_comes_from_table_not_air():
    c = Codec(KEY)
    t = PeerTable(c, anchors={0: AnchorInfo(1.0, 2.0, 1.5, 0.03)})
    # an attacker (with the key, i.e. an insider) claims to be anchor 0 at a false place
    t.ingest(c.encode(snap(node=0, pos=(99.0, 99.0), anchor=True), 1), 0.0)
    o = t.observe(0, 5.0, 0.0)
    assert o.is_anchor and np.allclose(o.peer_pos, [1.0, 2.0])


def test_implausible_rejected():
    c = Codec(KEY)
    t = PeerTable(c, max_speed=10.0)
    assert t.ingest(c.encode(snap(vel=(30.0, 0.0)), 1), 0.0) is None and t.rejected["implausible"] == 1


def test_decoder_never_raises_on_garbage():
    rng = np.random.default_rng(0)
    c = Codec(KEY)
    for n in (0, 1, 5, 31, 32, 33, 64):
        for _ in range(200):
            assert c.decode(bytes(rng.integers(0, 256, n, dtype=np.uint8))) is None
