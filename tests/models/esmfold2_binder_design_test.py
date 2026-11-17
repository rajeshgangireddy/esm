import inspect
import math
from types import SimpleNamespace
from unittest.mock import patch

import biotite.structure as struc
import torch

from esm.models.esmfold2 import binder_design as bd


def test_build_atom_array_and_build_complex_return_valid_structures() -> None:
    inputs = {
        "atom_to_token": torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
        "res_type": torch.tensor([[2, 2]], dtype=torch.long),
        "token_index": torch.tensor([[0, 1]], dtype=torch.long),
        "asym_id": torch.tensor([[0, 1]], dtype=torch.long),
        "mol_type": torch.tensor([[0, 0]], dtype=torch.long),
        "ref_atom_name_chars": torch.zeros((1, 4, 4), dtype=torch.long),
        "ref_element": torch.zeros((1, 4), dtype=torch.long),
        "atom_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.bool),
    }
    for atom_idx, name in enumerate(["N", "CA", "N", "CA"]):
        for char_idx, ch in enumerate(name):
            inputs["ref_atom_name_chars"][0, atom_idx, char_idx] = ord(ch) - 32
        inputs["ref_element"][0, atom_idx] = 6 if name == "N" else 5
    output = {
        "sample_atom_coords": torch.zeros((1, 4, 3), dtype=torch.float32),
        "plddt_per_atom": torch.full((1, 4), 0.85, dtype=torch.float32),
    }

    atom_arr = bd.build_atom_array(inputs, output)
    assert isinstance(atom_arr, struc.AtomArray)
    assert len(atom_arr) == 4
    assert list(dict.fromkeys(atom_arr.chain_id)) == ["A", "B"]

    protein_complex = bd.build_complex(inputs, output, atom_arr=atom_arr)
    assert len(list(protein_complex.chain_iter())) == 2


def test_compute_distogram_iptm_proxy_handles_missing_abnumber() -> None:
    logits = torch.zeros((6, 6, 128), dtype=torch.float32)
    with patch.object(bd, "_cdr_indices", side_effect=ImportError("no abnumber")):
        scores = bd.compute_distogram_iptm_proxy(
            logits, target_length=3, binder_sequence="AAA", is_antibody=True
        )
    assert math.isfinite(scores["distogram_iptm_proxy"])
    assert math.isnan(scores["cdr_distogram_iptm_proxy"])


def test_design_binder_accepts_num_steps_and_homotypic_chains() -> None:
    sig = inspect.signature(bd.design_binder)
    assert "num_steps" in sig.parameters
    assert sig.parameters["num_steps"].default == bd.STEPS

    recorded: dict[str, list[object]] = {}

    def _prepare(inputs_raw, max_atoms=None):
        recorded["sequences"] = list(inputs_raw.sequences)
        return {
            "res_type": torch.zeros((9,), dtype=torch.long),
            "token_index": torch.arange(9, dtype=torch.long),
        }

    dummy_model = SimpleNamespace(
        __call__=lambda self, **_kw: None,
    )

    class _DummyModel:
        def __call__(self, **_kw):
            return {
                "distogram_logits": torch.zeros((1, 9, 9, 128)),
                "sample_atom_coords": torch.zeros((1, 9, 3)),
            }

    design = torch.zeros((1, 3, 20))
    design[:, :, 0] = 1.0  # 'A'
    with patch.object(bd, "prepare_esmfold2_tensors", _prepare):
        bd.fold_and_get_distogram(
            _DummyModel(),  # type: ignore[arg-type]
            target_seq="AAA",
            target_one_hot=torch.zeros((1, 3, 33)),
            design=design,
        )
    assert len(recorded["sequences"]) == 2
    assert [s.id for s in recorded["sequences"]] == ["0", "1"]  # type: ignore[attr-defined]
    assert [s.sequence for s in recorded["sequences"]] == ["AAA", "AAA"]  # type: ignore[attr-defined]


def test_compute_esmc_pseudoperplexity_nll_casts_on_cpu_and_uses_device_autocast() -> None:
    autocast_calls: list[tuple[str, torch.dtype, bool]] = []
    orig_autocast = torch.autocast

    def _spy_autocast(device_type: str, dtype: torch.dtype = torch.bfloat16, enabled: bool = True):
        autocast_calls.append((device_type, dtype, enabled))
        return orig_autocast(device_type=device_type, dtype=dtype, enabled=enabled)

    vocab_size = 64
    hidden_dim = 8
    embed = torch.nn.Embedding(vocab_size, hidden_dim, dtype=torch.float32)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, dtype=torch.float32)
    esmc = SimpleNamespace(
        embed=embed,
        transformer=lambda x, **_kw: (torch.zeros((*x.shape[:2], hidden_dim), dtype=torch.float32),),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=vocab_size, mask_token_id=32),
        esmc=esmc,
        lm_head=lm_head,
    )
    binder = torch.softmax(torch.randn((1, 4, 20)), dim=-1)
    mask = torch.ones((1, 4), dtype=torch.bool)
    with patch.object(torch, "autocast", _spy_autocast):
        nll = bd.compute_esmc_pseudoperplexity_nll(
            model, binder, mask, batch_size=2, n_passes=2  # type: ignore[arg-type]
        )
    assert nll.shape == (1,)
    assert torch.isfinite(nll).all()
    assert autocast_calls == [("cpu", torch.bfloat16, False)]


if __name__ == "__main__":
    test_build_atom_array_and_build_complex_return_valid_structures()
    test_compute_distogram_iptm_proxy_handles_missing_abnumber()
    test_design_binder_accepts_num_steps_and_homotypic_chains()
    test_compute_esmc_pseudoperplexity_nll_casts_on_cpu_and_uses_device_autocast()
    print("ALL 4 PASSED")
