"""Apply the de-AI copy-edit pass to the CARC paper and supplementary .docx.

Style-only edits: no number, statistic, record name, citation, or cross-reference
changes. Each entry is (old, new); `old` must appear exactly once in the file,
which is asserted so a silently-missed edit fails loudly.

Plan of record: outputs/deai_edit_plan.md (two blocks overridden here -- see
MAIN entries flagged "override").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from docx import Document  # noqa: E402
from docx_edit import replace_everywhere  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

MAIN = [
    # --- 1. Introduction ---
    ("Questions of this kind—whether Greenland warming preceded Antarctic warming, or whether changes in atmospheric CO₂ led or lagged changes in temperature—have motivated",
     "Questions of this kind, whether Greenland warming preceded Antarctic warming or whether changes in atmospheric CO₂ led or lagged changes in temperature, have motivated"),
    ("whether additional structural factors—independent of data quality—further constrain causal discovery",
     "whether additional structural factors, independent of data quality, further constrain causal discovery"),
    ("Taken together, these three factors raise a practical question.",
     "These three factors together raise a practical question."),
    ("bounded below by approximately 35%—seven times the nominal 5% level.",
     "bounded below by approximately 35%, seven times the nominal 5% level."),
    ("by constructing a diagnostic framework rather than by proposing yet another lag estimate.",
     "by constructing a diagnostic framework instead of proposing yet another lag estimate."),
    ("may be replaced by CCM, PCMCI, or other method—so that CARC functions as a general robustness check rather than as a single fixed procedure.",
     "may be replaced by CCM, PCMCI, or other method; CARC then functions as a general robustness check, not as a single fixed procedure."),
    ("indicating that the obstacle is data quality—signal-to-noise ratio plus chronology error—which lies below",
     "indicating that the obstacle is data quality (signal-to-noise ratio plus chronology error), which lies below"),
    ("a quantitative and transferable benchmarking procedure, a form of subtractive inquiry that maps out what cannot reliably be known",
     "a quantitative and transferable benchmarking procedure that maps out what cannot reliably be known"),
    ("dataset of 80 proxy records—ice cores, marine sediments, loess, and speleothems—spanning the North Atlantic",
     "dataset of 80 proxy records (ice cores, marine sediments, loess, and speleothems) spanning the North Atlantic"),

    # --- 2. Theoretical framework ---
    ("responses to a common driver—such as North Atlantic freshwater flux or atmospheric CO₂—is essential",
     "responses to a common driver, such as North Atlantic freshwater flux or atmospheric CO₂, is essential"),
    ("one specific source of spurious association—shared external forcing—and does not address",
     "one specific source of spurious association, shared external forcing, and does not address"),
    ("To see why the problem is structural rather than incidental, consider",
     "To see why the problem is structural, not incidental, consider"),
    ("The physical intuition is as follows.", "The physical intuition is simple."),
    ("Theorem 2 answers this by deriving the expected F-statistic in closed form.",
     "Theorem 2 derives the expected F-statistic in closed form."),
    # override: the plan kept the rhetorical question, which is the tell itself
    ("in terms of the underlying parameters—autoregressive coefficients, forcing loadings, and the persistence of the common driver. For the paleoclimate practitioner, however, what ultimately matters is not the expected F-value per se but the false-positive rate that results from it. How often would a researcher using standard Granger causality on untreated proxy data erroneously conclude that X causes Y? Theorem 3 translates the F-statistic inflation into a lower bound on this rate, and the accompanying numerical evaluation shows that the bound is far from academic.",
     "in terms of the underlying parameters: autoregressive coefficients, forcing loadings, and the persistence of the common driver. What matters in practice, however, is the false-positive rate that results from this inflation, not the expected F-value itself. Theorem 3 translates the inflation into a lower bound on that rate, and the numerical evaluation below shows the bound to be substantial."),
    ("might overstate the false-positive risk and therefore paint an unduly pessimistic picture. We now show the opposite: the linear bound is conservative, and under the steep-response dynamics",
     "might overstate the false-positive risk and therefore look unduly pessimistic. The linear bound is conservative instead: under the steep-response dynamics"),
    ("We now show that this omission does not weaken the argument—on the contrary,",
     "This omission does not weaken the argument—on the contrary,"),
    ("—a condition almost always met for the steep threshold responses characteristic of D–O and Heinrich events—",
     ", a condition almost always met for the steep threshold responses characteristic of D–O and Heinrich events,"),
    ("Taken together, Theorems 1-4 identify three practical requirements",
     "Theorems 1-4 together identify three practical requirements"),

    # --- 3. Methods ---
    ("chronological uncertainty must be explicitly propagated—Theorems 1–3 assume error-free age labels",
     "chronological uncertainty must be explicitly propagated: Theorems 1–3 assume error-free age labels"),
    ("under persistent common forcing—non-zero residual–regressor covariance (Theorem 1)",
     "under persistent common forcing: non-zero residual–regressor covariance (Theorem 1)"),
    ("ensembles of plausible chronologies rather than point estimates,",
     "ensembles of plausible chronologies instead of point estimates,"),
    ("into the causal inference rather than treating it as a nuisance parameter.",
     "into the causal inference instead of treating it as a nuisance parameter."),
    ("attenuates the autocorrelation structure—the autoregressive (AR) coefficient drops from about 0.8 to about 0.1—potentially introducing",
     "attenuates the autocorrelation structure (the autoregressive (AR) coefficient drops from about 0.8 to about 0.1), potentially introducing"),
    ("Rigorously propagate chronological uncertainty rather than relying on a single best age scale.",
     "Rigorously propagate chronological uncertainty instead of relying on a single best age scale."),
    ("in the opposite direction—on synthetic benchmarks, the true-edge detection rate divided by the negative-control false-positive rate; on real records, the forward robustness score divided by the reverse robustness score—with the denominator floored",
     "in the opposite direction (on synthetic benchmarks, the true-edge detection rate divided by the negative-control false-positive rate; on real records, the forward robustness score divided by the reverse robustness score), with the denominator floored"),

    # --- 4. Results ---
    ("cannot recover the prescribed edges; it can only admit the common-forcing artifacts",
     "cannot recover the prescribed edges; it admits only the common-forcing artifacts"),
    # override: plan's wording collided with the preceding "nevertheless"
    ("though its pointwise errors are nevertheless officially published—and all six carry traceable age uncertainties",
     "though its pointwise errors are officially published. All six therefore carry traceable age uncertainties"),
    ("Three diagnostic patterns emerge.", "Three patterns emerge from these results."),
    ("This near-symmetry is the statistical fingerprint of the common-forcing dominance",
     "This near-symmetry is the statistical signature of the common-forcing dominance"),
    ("and GS-12 windows require interpretation. Both windows are narrow",
     "and GS-12 windows reflect limited statistical power, not the absence of coupling. Both windows are narrow"),
    ("to evaluate the pairing—it is a statement about data adequacy, not about the absence of causal coupling.",
     "to evaluate the pairing; the score reflects data adequacy, not the absence of causal coupling."),
    ("remove one layer at a time—observational noise, age perturbation, or deconfounding.",
     "remove one layer at a time: observational noise, age perturbation, or deconfounding."),
    ("The raw output is generous: 75 of the 80 records pass",
     "The raw screen is permissive: 75 of the 80 records pass"),
    ("under forced differencing—the honesty check that removes the smooth low-frequency component the ADF gate can miss—leaving 0 of the 6,320",
     "under forced differencing (the honesty check that removes the smooth low-frequency component the ADF gate can miss), leaving 0 of the 6,320"),
    ("locates the binding constraints—removing observational noise recovers a mean of only 1.6 edges",
     "locates the binding constraints: removing observational noise recovers a mean of only 1.6 edges"),
    ("the shared low-frequency common forcing of Section 2 rather than record-specific",
     "the shared low-frequency common forcing of Section 2, not record-specific"),
    ("the statistical signature of common forcing rather than of directional dominance.",
     "the statistical signature of common forcing, not of directional dominance."),
    ("Records whose raw passes are pure trend leakage—for example Lake Junín (87 passes), U1302-U1303  (86), and Dome Fuji D (84)—carry no event-scale",
     "Records whose raw passes are pure trend leakage, for example Lake Junín (87 passes), U1302-U1303  (86), and Dome Fuji D (84), carry no event-scale"),
    ("Conversely, the records recovering the most edges once age errors are removed—U1302-U1303 Ca/Sr (31), MD01-2443 alkenone SST (29), and Lake Elgygytgyn Si/Ti (27)—are the targets on which improved dating is most likely to unlock inference",
     "Conversely, improved dating is most likely to enable inference for the records recovering the most edges once age errors are removed: U1302-U1303 Ca/Sr (31), MD01-2443 alkenone SST (29), and Lake Elgygytgyn Si/Ti (27)"),
    ("every pass collapses under forced differencing—the honesty check admits 0 of the 6,320 directed pairs in all 80 records.",
     "every pass collapses under forced differencing; the honesty check admits 0 of the 6,320 directed pairs in all 80 records."),
    ("when one layer—observational noise, age error, or ARC deconfounding—is removed",
     "when one layer, observational noise, age error, or ARC deconfounding, is removed"),
    ("indicating common forcing rather than directional dominance.",
     "indicating common forcing, not directional dominance."),
    ("sharpens directional asymmetry—as designed—but does not by itself produce",
     "sharpens directional asymmetry, as designed, but does not by itself produce"),
    ("mark candidate reproducible relationships rather than confirmed physical mechanisms:",
     "mark candidate reproducible relationships, not confirmed physical mechanisms:"),
    ("which makes genuine leads harder—not easier—to detect.",
     "which makes genuine leads harder, not easier, to detect."),
    ("these temporal tendencies are not causal directions but candidate relationships for future testing with higher-precision data.",
     "these temporal tendencies are candidate relationships for future testing with higher-precision data, not causal directions."),

    # --- 5. Discussion / 6. Conclusions ---
    ("This outcome cannot be blamed on a particular method or record; it reflects the combined effect",
     "This outcome is not specific to any particular method or record; it reflects the combined effect"),
    ("The term is operational. It describes a state of the current data quality",
     "The term is operational: it describes a state of the current data quality"),
    ("through Monte Carlo perturbation rather than an analytical ratio",
     "through Monte Carlo perturbation in place of an analytical ratio"),
    ("for FPR inflation—equation (5)—attributing", "for FPR inflation (equation (5)), attributing"),
    ("rather than to a generic method limitation.", "and not to a generic method limitation."),
    ("This does not make them wrong; it only means they were obtained under a different statistical standard.",
     "This does not make them wrong; the estimates were obtained under a different statistical standard."),
    ("compression alone is therefore insufficient—the residual gap reflects",
     "compression alone is therefore insufficient; the residual gap reflects"),
    ("inflates the worst control to 54.5%—chronological precision alone cannot push",
     "inflates the worst control to 54.5%: chronological precision alone cannot push"),
    ("real pairings projected on the escape plane—six-record pairs (×, H1 emphasized ◆)",
     "real pairings projected on the escape plane: six-record pairs (×, H1 emphasized ◆)"),
    ("Several limitations should be noted.", "The framework has several limitations."),
    ("making true directional signals harder—not easier—to detect.",
     "making true directional signals harder, not easier, to detect."),
    ("are statistical lead-lag relationships rather than structural physical causality",
     "are statistical lead-lag relationships, not structural physical causality"),
    ("about 35%—seven times the nominal 5% level (Theorems 1–3)—and Theorem 4 proves",
     "about 35%, seven times the nominal 5% level (Theorems 1–3), and Theorem 4 proves"),
    ("results—reliable detection in the benchmarks before chronology error is applied, but sub-threshold performance once it is, exactly as on the records—shows that the limiting factor is data quality rather than methodology:",
     "results (reliable detection in the benchmarks before chronology error is applied, but sub-threshold performance once it is, exactly as on the records) shows that the limiting factor is data quality, not methodology:"),
]

SUPP = [
    ("guided by climate dynamics — North Atlantic (Greenland, subpolar North Atlantic cores)",
     "guided by climate dynamics: North Atlantic (Greenland, subpolar North Atlantic cores)"),
    ("when one layer—observational noise, age perturbation, or ARC deconfounding—is removed.",
     "when one layer, observational noise, age perturbation, or ARC deconfounding, is removed."),
    ("The repository reconstruction—unconditional first-differencing, AIC lag selection 1–5, χ² test at the 0.05 level, the NGRIP series left undeconfounded, 5% observation noise added to each perturbation member, 200 draws—lies systematically",
     "The repository reconstruction (unconditional first-differencing, AIC lag selection 1–5, χ² test at the 0.05 level, the NGRIP series left undeconfounded, 5% observation noise added to each perturbation member, 200 draws) lies systematically"),
]


def apply(path, edits, label):
    doc = Document(path)
    failed = []
    for old, new in edits:
        try:
            replace_everywhere(doc, old, new, expect=1)
        except (AssertionError, ValueError) as exc:
            failed.append((old, exc))
    doc.save(path)
    print(f"{label}: {len(edits) - len(failed)}/{len(edits)} applied")
    for old, exc in failed:
        print(f"  FAILED  {old[:80]!r}\n          {exc}")
    return failed


if __name__ == "__main__":
    bad = apply(ROOT / "paper" / "Research Pape5.docx", MAIN, "main text")
    bad += apply(ROOT / "paper" / "CARC_supplementary_edited.docx", SUPP, "supplementary")
    sys.exit(1 if bad else 0)
