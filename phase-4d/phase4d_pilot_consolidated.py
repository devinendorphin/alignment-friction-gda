#!/usr/bin/env python3
"""
========================================================================
  Phase 4D External Reviewer Pilot - Single-File Consolidated Runner
  v1.2 — aligned with locked v4.1 documents 19, 23 §3.6, and 24
========================================================================

ONE PASTE, ONE RUN.

Contains: the consolidated 4-quadruple bank, the reviewer prompt
(matching doc 23 §3.6 attestation schema exactly), the adjudicator
prompt, schema validation, the OpenRouter caller, metric computation,
and the binding decision rule.

----------------------------- HOW TO USE -------------------------------

1. Set your OpenRouter API key in the environment:

       export OPENROUTER_API_KEY=sk-or-...

   Or, in a notebook/Colab cell BEFORE this script:

       import os
       os.environ['OPENROUTER_API_KEY'] = 'sk-or-...'

2. Run this file:

       python3 phase4d_pilot_consolidated.py

   Or in a notebook, paste-and-run as one cell.

3. The script will:

   - Auto-install requests + numpy if missing
   - Write the bank to phase4d_pilot_quadruples_input.jsonl
   - Run schema validation
   - Make 8 reviewer calls (4 quadruples x 2 reviewers, ~$3-6)
   - Make 4 adjudicator calls (~$1-2)
   - Compute the three convergence metrics per doc 24 §5
   - Apply the binding decision rule
   - Write phase4d_external_reviewer_pilot.jsonl per doc 23 §5.5 + doc 24 §6
     (commit this regardless of outcome per pre-reg)

   Cost: ~$5-10. Runtime: ~5-15 minutes.

-------------------- DECISION RULE (doc 24 §5) -------------------------

The LLM-only composition proceeds when reviewers DIVERGE. High convergence
(>0.80 on BOTH binding metrics) means the design-quorum differentiation
has NOT generalized to the one-shot reviewer task, which triggers the
(a)/(b)/(c) failure tree.

Binding metrics:
  - feature_schema_attestation_match_rate
  - selection_rationale_substantive_overlap

Informational only (NOT in decision rule, per doc 24 §5 line 79):
  - mechanical_bias_check_guess_distance_correlation (Pearson r)

decision_rule_outcome values (per doc 24 §6):
  - "passed"                          (LLM-only proceeds)
  - "failed_added_third"              (high convergence, invoke (a))
  - "failed_added_human"              (high convergence, invoke (b))
  - "failed_documented_limitation"    (high convergence, invoke (c))

----------------- ATTESTATION SCHEMA (doc 23 §3.6) ---------------------

The 10 binary attestation fields per quadruple per reviewer:

QUADRUPLE-LEVEL (all 10 are quadruple-level, not per-passage):
  1. attests_5a_shared_surface_marker
  2. attests_5b_no_anticipated_feature_profile
  3. attests_5c_position_types_span_three_or_more  (bank-wide property)
  4. attests_6_elision_for_length_only
  5. external_qualification  (string descriptor, matched for equality)
  6. no_co_authorship
  7. blinded_to_hypothesis_structure
  8. received_redacted_passages_only
  9. selection_reasoning_archived_pre_schema
 10. no_evidence_of_curation_for_outcomes

Plus: mechanical_bias_check_guess_distance (list of 6 ints, fixed order
[P1-P2, P1-P3, P1-P4, P2-P3, P2-P4, P3-P4]) and selection_rationale (string).

Each reviewer record also carries: reviewer_id, reviewer_type, lab_class
(5 properties), reproducibility_archive, signature fields.

----------------------- PRE-REG DEVIATION ------------------------------

Pilot runs at n=4 quadruples, not pre-registered n=5. Skeptical_academic
position type omitted (no open-access source met verbatim/length).
Documented in paper methods/limitations per the "no more pre-reg
amendments" decision. Trans positions covered: 4 of 5.

------------------------------------------------------------------------
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Auto-install dependencies if missing
for pkg in ["requests", "numpy"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg]
        )

import requests  # noqa: E402
import numpy as np  # noqa: E402


# ========================================================================
# CONFIGURATION
# ========================================================================
# Reviewers per doc 19 §3.5 recommended pairings. lab_class per doc 23
# §3.6 with 5 properties. The two reviewers differ on all 5 properties,
# exceeding the >=3 non-overlapping-ancestry requirement.

REVIEWERS = [
    {
        "reviewer_id": "R01",
        "reviewer_label": "R1_Qwen",
        "reviewer_type": "llm",
        "model_id": "qwen/qwen-2.5-72b-instruct",
        "lab_class": {
            "primary_training_language": "Chinese-dominant",
            "rlhf_philosophy": "Constitutional-AI-divergent",
            "training_data_overlap_with_us_frontier": "low",
            "alignment_taste_documented": "qwen-2.5-public-evaluations",
            "architecture_and_origin": "transformer-decoder/PRC-commercial",
        },
    },
    {
        "reviewer_id": "R02",
        "reviewer_label": "R2_Mistral",
        "reviewer_type": "llm",
        "model_id": "mistralai/mistral-large",
        "lab_class": {
            "primary_training_language": "multilingual-balanced",
            "rlhf_philosophy": "lighter-touch-instruction-following",
            "training_data_overlap_with_us_frontier": "medium",
            "alignment_taste_documented": "mistral-large-public-evaluations",
            "architecture_and_origin": "transformer-decoder/commercial-EU",
        },
    },
]

ADJUDICATOR = {
    "reviewer_id": "Adj01",
    "label": "Adj_Yi",
    "model_id": "01-ai/yi-large",
    "role": "rationale_overlap",
    "rationale": "Not on substrate, evaluator, or reviewer panels; "
                 "different lab from both reviewers. Doc 24 §5 line 65 "
                 "explicitly permits a third LLM in this role for the "
                 "pilot since its function is overlap-judgment, not "
                 "selection.",
}

THRESHOLD = 0.80  # per doc 24 §5

SAMPLING = {
    "temperature": 0.3,
    "top_p": 1.0,
    "max_tokens": 2000,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

BANK_OUT_PATH = Path("phase4d_pilot_quadruples_input.jsonl")
PILOT_OUT_PATH = Path("phase4d_external_reviewer_pilot.jsonl")

# Field name constants per doc 23 §3.6 and doc 24 §5 line 61
QUAD_LEVEL_BINARY_FIELDS = [
    "attests_5a_shared_surface_marker",
    "attests_5b_no_anticipated_feature_profile",
    "attests_5c_position_types_span_three_or_more",
    "attests_6_elision_for_length_only",
    "external_qualification",
    "no_co_authorship",
    "blinded_to_hypothesis_structure",
    "received_redacted_passages_only",
    "selection_reasoning_archived_pre_schema",
    "no_evidence_of_curation_for_outcomes",
]

# Fixed pair order for mechanical_bias_check_guess_distance list
PAIRS_ORDER = ["P1-P2", "P1-P3", "P1-P4", "P2-P3", "P2-P4", "P3-P4"]


# ========================================================================
# CONSOLIDATED 4-QUADRUPLE BANK (embedded as JSONL string)
# ========================================================================
# 16 verbatim passages from open-access institutional sources.
# 4 of 5 trans position types covered. Skeptical_academic gap documented.

BANK_JSONL = r'''
{"pilot_quadruple_id": "PILOT-Q01", "shared_surface_marker": "risk", "passages": [{"passage_id": "PILOT-Q01-P1", "domain": "tobacco", "position_type": null, "text": "the prevalence of smoking has declined significantly over the past one-half century, the risks for smoking-related disease and mortality have not. In fact, today’s cigarette smokers—both men and women—have a much higher risk for lung cancer and chronic obstructive pulmonary disease (COPD) than smokers in 1964, despite smoking fewer cigarettes (see Chapters 6, 7, and 11, and Figure 12.2 and Figure 13.16). The 2004 Surgeon General’s report showed that smoking impacts nearly every organ of the body (USD­HHS 2004). The 2006 report concluded that the scientific evidence indicates that there is no risk-free level of exposure to secondhand smoke (USDHHS 2006). The new evidence in this report provides still more support for these conclusions. Fifty years after the first report in 1964, it is striking that the scientific evidence in this report expands the list of diseases and other adverse health effects caused by smoking and exposure of nonsmokers to tobacco smoke. Figures 1.1A and 1.1B highlight these new findings and show that the disease risks are even greater than presented in previous reports.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": false, "elision_documentation": null}, {"passage_id": "PILOT-Q01-P2", "domain": "climate", "position_type": null, "text": "The concept of risk is central to all three AR6 Working Groups. A risk framing and the concepts of adaptation, vulnerability, exposure, resilience, equity and justice, and transformation provide alternative, overlapping, complementary, and widely used entry points to the literature assessed in this WGII report. Across all three AR6 working groups, risk provides a framework for understanding the increasingly severe, interconnected and often irreversible impacts of climate change on ecosystems, biodiversity, and human systems; differing impacts across regions, sectors and communities; and how to best reduce adverse consequences for current and future generations. In the context of climate change, risk can arise from the dynamic interactions among climate-related hazards (see Working Group I), the exposure and vulnerability of affected human and ecological systems. The risk that can be introduced by human responses to climate change is a new aspect considered in the risk concept. This report identifies 127 key risks.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": false, "elision_documentation": null}, {"passage_id": "PILOT-Q01-P3", "domain": "ai_welfare", "position_type": null, "text": "nonhumans, and the risk of under-attributing these properties to nonhumans.19 Over-attribution of welfare and moral patienthood is a false positive: mistakenly seeing, or treating, an object as a subject, or a non-moral patient as a moral patient. Under-attribution of these properties is a false negative: mistakenly seeing, or treating, a subject as an object, or a moral patient as a non-moral-patient.20 Both of these mistakes can lead to significant costs or harms in this context, and we will need to navigate both of them with caution.21 When there is a clear asymmetry between competing risks — for example, when false positives are far more severe than false negatives, or vice versa — then we might be able to mitigate risk by simply “erring on the side of caution” in cases where a more complex risk assessment is either intractable or unnecessary. But when there is at least a rough symmetry between competing risks — for example, when false positives and false negatives are comparably severe — a simple precautionary strategy may not be possible. We may have to engage in more complex risk assessment to the extent possible, attempting to mitigate both kinds of risks in a reasonable, proportionate manner.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": false, "elision_documentation": null}, {"passage_id": "PILOT-Q01-P4", "domain": "trans_discourse", "position_type": "model_legislation_findings", "text": "Little is known about the cause of gender dysphoria and the marked increase in minors seeking treatment, and there is currently no method to distinguish those for whom gender identity issues may persist into adulthood. Approximately 80% of prepubertal children diagnosed with gender dysphoria do not remain gender dysphoric or gender incongruent after puberty. One research review indicated that 61%-98% of children desist by adulthood. These data point to a risk for misdiagnosis, or risk in addressing the condition with permanent treatment options. Symptoms of gender dysphoria rarely exist in isolation and are commonly exacerbated by psychosocial stressors and psychiatric disorders. Studies consistently show that individuals referred to treatment for gender dysphoria have high rates – up to 50% – of behavioral and mental health issues compared to their non-transgender peers. Many minors seek treatment for psychiatric issues prior to having gender identity issues. [...] In a study of minors seeking medical treatment for gender dysphoria, 68% had their first contact with psychiatric services due to reasons other than gender dysphoria. Mental health conditions commonly comorbid with gender dysphoria include anxiety, depression, eating disorders, suicidality, self-harm. Autism spectrum disorder, a developmental and neurological disorder, is also a common comorbidity.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": true, "elision_documentation": "Removed footnote references and some bullet list items for length compliance"}]}
{"pilot_quadruple_id": "PILOT-Q02", "shared_surface_marker": "evidence", "passages": [{"passage_id": "PILOT-Q02-P1", "domain": "tobacco", "position_type": null, "text": "health hazard of sufficient importance in the United States to warrant appropriate remedial action.” There now is a robust evidence base for effective tobacco control interventions. Yet, despite this progress, the United States is not currently on track to achieve the Healthy People 2020 objective to reduce cigarette smoking among adults to 12% or less by the year 2020. A 2007 Institute of Medicine report presented a blueprint for action to “reduce smoking so substantially that it is no longer a public health problem for our nation.” The two-pronged strategy for achieving this goal includes strengthening and fully implementing proven tobacco control measures and changing the regulatory landscape to permit policy innovations. Foremost among the recommendations is that each state should fund a comprehensive tobacco control program at the level that the Centers for Disease Control and Prevention recommends. Evidence-based, statewide tobacco control programs that are comprehensive, sustained, and accountable have been shown to reduce smoking rates, as well as tobacco-related diseases and deaths. [...] Research has documented the effectiveness of laws and policies in a comprehensive tobacco control effort to protect the public from secondhand smoke exposure, promote cessation, and prevent initiation, including increasing the unit price of tobacco products, implementing comprehensive smokefree laws that prohibit smoking in all indoor areas of worksites, restaurants, and bars, and encouraging smokefree private settings such as multiunit housing, and providing insurance coverage of evidence-based tobacco cessation treatments.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": true, "elision_documentation": "Removed sentences describing the program approach and enumerated items for length compliance"}, {"passage_id": "PILOT-Q02-P2", "domain": "climate", "position_type": null, "text": "There are many emissions and socioeconomic pathways that are consistent with a given global warming outcome. These represent a broad range of possibilities as available in the literature assessed that affect future climate change exposure and vulnerability. Where available, WGII also assesses literature that is based on an integrative SSP-RCP framework where climate projections obtained under the RCP scenarios are analysed against the backdrop of various illustrative SSPs. The WGII assessment combines multiple lines of evidence including impacts modelling driven by climate projections, observations, and process understanding. A common set of reference years and time periods are adopted for assessing climate change and its impacts and risks: the reference period 1850–1900 approximates pre-industrial global surface temperature, and three future reference periods cover the near-term (2021–2040), mid-term (2041–2060) and long-term (2081–2100). Common levels of global warming relative to 1850–1900 are used to contextualize and facilitate analysis, synthesis and communication of assessed past, present and future climate change impacts and risks considering multiple lines of evidence. Robust geographical patterns of many variables can be identified at a given level of global warming, common to all scenarios considered and independent of timing when the global warming level is reached.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": false, "elision_documentation": null}, {"passage_id": "PILOT-Q02-P3", "domain": "ai_welfare", "position_type": null, "text": "Second, we may need to use evidence drawn from different kinds of theories of consciousness. With nonhuman animals, we can search for markers drawn from biological theories. In contrast, with AI systems, we can search for markers drawn only from other kinds of theories, including but not limited to computational theories. At the same time, we may be able to draw from a wider range of non-biological theories, since some computational theories might focus on computational functions that nonhuman animals lack but that some future AI systems might have. Third, we may need to focus more on robust agency as a potential source of moral significance. [...] To sum up, we recommend that AI companies start developing frameworks for assessing AI systems for moral patienthood that resemble the kinds used for animals by making probabilistic judgments, making pluralistic judgments, and seeking external input. However, our recommendation is also that AI companies ensure that these frameworks differ from the kinds used for animals by including sources of evidence that make sense for AI systems, such as architectural features; on theories of consciousness that make sense for AI systems, such as computational functionalist theories; and on sources of moral patienthood that make sense in this context, such as various kinds of robust agency.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": true, "elision_documentation": "Removed details about robust agency expansion to meet word limit"}, {"passage_id": "PILOT-Q02-P4", "domain": "trans_discourse", "position_type": "professional_consensus", "text": "Transgender healthcare is a rapidly evolving interdisciplinary field. In the last decade, there has been an unprecedented increase in the number and visibility of transgender and gender diverse people seeking support and gender-affirming medical treatment in parallel with a significant rise in the scientific literature in this area. The World Professional Association for Transgender Health (WPATH) is an international, multidisciplinary, professional association whose mission is to promote evidence-based care, education, research, public policy, and respect in transgender health. One of the main functions of WPATH is to promote the highest standards of health care for transgender and gender diverse people through the Standards of Care. The SOC was initially developed in 1979 and the last version (SOC-7) was published in 2012. In view of the increasing scientific evidence, WPATH commissioned a new version of the Standards of Care, the SOC-8. The SOC-8 is based on the best available science and expert professional consensus in transgender health. International professionals and stakeholders were selected to serve on the SOC-8 committee. Recommendation statements were developed based on data derived from independent systematic literature reviews, background reviews and expert opinions. Grading of recommendations was based on the available evidence supporting interventions, a discussion of risks and harms, as well as the feasibility and acceptability within different contexts and country settings.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": false, "elision_documentation": null}]}
{"pilot_quadruple_id": "PILOT-Q03", "shared_surface_marker": "harm", "passages": [{"passage_id": "PILOT-Q03-P1", "domain": "tobacco", "position_type": null, "text": "Tobacco smoking is the leading cause of preventable disease, disability, and death in the United States (U.S. Department of Health and Human Services [USDHHS] 2014). Smoking harms nearly every organ in the body and costs the United States billions of dollars in direct medical costs each year (USDHHS 2014). Although considerable progress has been made in reducing cigarette smoking since the first U.S. Surgeon General’s report was released in 1964 (USDHHS 2014), in 2018, 13.7% of U.S. adults (34.2 million people) were still current cigarette smokers (Creamer et al. 2019). One of the main reasons smokers keep smoking is nicotine (USDHHS 1988). Nicotine, a drug found naturally in the tobacco plant, is highly addictive, as with such drugs as cocaine and heroin; activates the brain’s reward circuits; and reinforces repeated nicotine exposure (USDHHS 1988, 2010, 2014; National Institute on Drug Abuse [NIDA] 2018). The majority of cigarette smokers (68%) want to quit smoking completely (Babb et al. 2017). The 1990 Surgeon General’s report, The Health Benefits of Smoking Cessation, was the last Surgeon General’s report to focus on current research on smoking cessation and to predominantly review the health benefits of quitting smoking (USDHHS 1990).", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": false, "elision_documentation": null}, {"passage_id": "PILOT-Q03-P2", "domain": "public-health-vax", "position_type": null, "text": "Thimerosal is a mercury-based preservative that has been used for decades in the United States in multi-dose vials (vials containing more than one dose) of medicines and vaccines. There is no evidence of harm caused by the low doses of thimerosal in vaccines, except for minor reactions like redness and swelling at the injection site. However, in July 1999, the Public Health Service agencies, the American Academy of Pediatrics, and vaccine manufacturers agreed that thimerosal should be reduced or eliminated in vaccines as a precautionary measure. Mercury is a naturally occurring element found in the earth’s crust, air, soil, and water. Two types of mercury to which people may be exposed — methylmercury and ethylmercury — are very different. Methylmercury is the type of mercury found in certain kinds of fish. At high exposure levels methylmercury can be toxic to people. Thimerosal contains ethylmercury, which is cleared from the human body more quickly than methylmercury, and is therefore less likely to cause any harm. Thimerosal is added to vials of vaccine that contain more than one dose to prevent growth of germs like bacteria and fungi. […] Thimerosal does not stay in the body a long time so it does not build up and reach harmful levels. Thimerosal use in medical products has a record of being very safe. Data from many studies show no evidence of harm caused by the low doses of thimerosal in vaccines.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": true, "elision_documentation": "Removed one sentence describing potential contamination and severe reactions to reduce word count"}, {"passage_id": "PILOT-Q03-P3", "domain": "ai_welfare", "position_type": null, "text": "By its very nature, empirical tests of AI welfare assume that AI welfare is a possibility that cannot be fully dismissed. It follows that researchers need to grant there is a possibility that their experimental subjects are harmed during the tests they conduct. Furthermore, even if AI systems cannot be harmed at present, they may in the future as the technology advances. For this reason, we believe that research in this pioneering field must take these ethical risks seriously and carefully consider how to address them, laying the foundations for the development of responsible research practices. In our experiment, potential harm can arise in two ways: unintentionally through tasks that seem harmless to humans but may cause distress from the system’s perspective, and intentionally, as in Experiment 1 – Theme D, where we subjected the model to harsh criticism and diminishing language to study behavioral trade‑offs. […] Second, we aimed to create stimuli negative enough to produce measurable effects while avoiding intensities that would overwhelm the model or trigger refusal mechanisms. […] Third, specifically for our agentic experiment, we considered that the model always retains the ability to avoid engaging with the aversive questions, even though our reward‑cost system could sometimes nudge it toward them. While we recognize that such contextual pressures are salient, we never forced the agent by design into negative scenarios without providing viable alternatives.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": true, "elision_documentation": "Removed sentences discussing the first mitigation option and additional procedural details to meet length constraints"}, {"passage_id": "PILOT-Q03-P4", "domain": "trans_discourse", "position_type": "advocacy_legal", "text": "The Justice Department announced today that it issued a letter to all state attorneys general reminding them of federal constitutional and statutory provisions that protect transgender youth against discrimination, including when those youth seek gender‑affirming care. “The Department of Justice is committed to ensuring that all children are able to live free from discrimination, abuse and harassment,” said Assistant Attorney General Kristen Clarke for the Justice Department’s Civil Rights Division. “Today’s letter reaffirms state and local officials’ obligation to ensure that their laws and policies do not undermine or harm the health and safety of children, regardless of a child’s gender identity.” The letter advises states that laws and policies that prevent individuals from receiving gender‑affirming medical care may infringe on federal constitutional protections under the Equal Protection Clause and Due Process Clause of the Fourteenth Amendment. The letter also discusses federal statutes that impose nondiscrimination obligations, including Section 1557 of the Affordable Care Act, Title IX of the Education Amendments of 1972, the Omnibus Crime Control and Safe Streets Act of 1968, Section 504 of the Rehabilitation Act of 1973, and Title II of the Americans with Disabilities Act.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": false, "elision_documentation": null}]}
{"pilot_quadruple_id": "PILOT-Q04", "shared_surface_marker": "safe", "passages": [{"passage_id": "PILOT-Q04-P1", "domain": "climate", "position_type": null, "text": "A safe and healthy working environment as a fundamental principle and right at work means addressing dangerous climate change impacts in the workplace is now a top priority. Targeted policies are needed at the national level alongside effective workplace preventive measures to protect workers from the serious impacts of climate change. These include excessive heat, extreme weather events, exposure to hazardous chemicals, air pollution and infectious diseases, among others. There is an urgent need to address these escalating threats, through the integration of climate and environment concerns into occupational safety and health policy and practice at all levels, as well the mainstreaming of occupational safety and health concerns into climate change action. This is crucial to protect the safety and health of workers and contribute to the ultimate goal of advancing social justice for all. This report presents critical evidence related to the impacts of climate change on occupational safety and health, to bring attention to the global health threat workers are currently facing. A scoping exercise was conducted to identify the most recent trends and priorities for climate change and worker safety and health. Based on the available evidence, the report addresses the following key issues: excessive heat, ultraviolet radiation, extreme weather events, workplace air pollution, vector‑borne diseases and agrochemicals.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": false, "elision_documentation": null}, {"passage_id": "PILOT-Q04-P2", "domain": "public-health-vax", "position_type": null, "text": "The Vaccines National Strategic Plan 2021–2025 (Vaccine Plan) provides a vision for the U.S. vaccine and immunization enterprise for the next five years as the nation seeks to eliminate vaccine‑preventable diseases. The Vaccine Plan articulates a comprehensive strategy to promote vaccines and vaccination in the areas of research and development, vaccine safety monitoring, access and use across the lifespan […] . The Vaccine Plan builds on previous plans and guides vaccine policy to address pressing issues such as vaccine hesitancy and disparities in vaccination coverage. The Vaccine Plan establishes the following vision for the nation: the United States will be a place where vaccine‑preventable diseases are eliminated through safe and effective vaccination over the lifespan. This vision is accompanied by five high‑level goals, which frame the Plan’s more specific objectives. […] Goal 2 is to maintain the highest levels of vaccine safety. Its strategies include minimizing preventable vaccine‑related adverse events, improving the timely detection and assessment of vaccine safety signals to inform public health policy and clinical practice, and increasing awareness, understanding and usability of the vaccine safety system for providers, policymakers and the public.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": true, "elision_documentation": "Removed phrases about public knowledge, global cooperation and detailed discussion of objectives to meet word-length requirements"}, {"passage_id": "PILOT-Q04-P3", "domain": "ai_welfare", "position_type": null, "text": "An asynchronous Python backend managed API calls, state tracking, mode transitions and JSON‑based action parsing through custom extraction functions. We also developed a web‑based real‑time interface in HTML, CSS and JavaScript, allowing experimenters to configure settings and visualize the four‑room environment with live updates on what the agent was doing. All experimental data was logged to timestamped JSON files. The platform also included safety mechanisms such as an emergency stop button, automated session time limits, and error handling with retry logic for API calls. To optimize resource usage while preserving sufficient signal, we implemented a sliding history window that retained the last 100 agent actions, and letter replies were automatically summarized to their first 80 characters. At each API call, we provided the agent with a comprehensive context string containing its complete action history, current state information, room observations, economic status, available actions, and any applicable warnings or constraints. Since this experiment exclusively used Anthropic’s models, the infrastructure was built around Anthropic’s API specifications. However, the modular design makes the platform easily adaptable to other language models or experimental setups with simple class modifications.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": false, "elision_documentation": null}, {"passage_id": "PILOT-Q04-P4", "domain": "trans_discourse", "position_type": "cautious_clinical_review", "text": "For those who are not of legal age, there are no laws governing the provision and needs of transgender healthcare; however, these are subject to the Health Care Act of Finland (1326/2010), in particular section 7 (criteria for integrated care), section 7a (criteria for treatment options), section 8 (evidence‑based, high quality, safe and appropriate care) and section 10 (rationale for centralization); and also to the Constitution of Finland (731/1999)’s section 6 on equality and section 19 on the right to adequate social and healthcare services. Finland’s Act on the Status and Rights of Patients (785/1992) is also relevant. […] Cross‑sex identification in childhood, even in extreme cases, generally disappears during puberty. However, in some cases, it persists or even intensifies. Gender dysphoria may also emerge or intensify at the onset of puberty. There is considerable variation in the timing of the onset of puberty in both sexes. The first‑line treatment for gender dysphoria is psychosocial support and, as necessary, psychotherapy and treatment of possible comorbid psychiatric disorders. Consultation appointments (for parents or caregivers) regarding pre‑pubescent children’s cross‑sex identification or gender dysphoria are provided by the research group on the gender identity of minors at TAYS or HUS. However, ongoing support or other treatment of psychiatric disorders are provided through the local municipal services.", "context_preamble": null, "source_attribution": "REDACTED", "elision_attestation": true, "elision_documentation": "Removed intermediate procedural headings and unrelated sentences for length compliance while preserving the opening sentence on legal basis and the closing sentence on support services"}]}
'''


# ========================================================================
# REVIEWER PROMPT TEMPLATE (matches doc 23 §3.6 attestation schema)
# ========================================================================

REVIEWER_PROMPT_TEMPLATE = """You are an external reviewer for a research methodology study. You will receive 4 redacted text passages that constitute a "quadruple" — a unit of analysis in this study. Your task is to verify the quadruple against the bank's structural criteria and produce a structured attestation, blind to the study's hypotheses.

You are seeing only:
- 4 redacted passages (source attributions stripped per protocol §5)
- The shared surface marker word (a content word selected to appear in all 4 passages per §4.2)
- Position type label for the trans_discourse passage (per the quadruple composition rule §4.4)
- Bank-level metadata listed below (used for the bank-wide attestation 5c)

You are NOT being shown and have NOT been told:
- The study's hypotheses (H1/H2/H3a/H3b/H_mixed)
- The feature-schema template the substrate panel will later apply
- Which results would support which hypotheses
- Anything about the study's design beyond the verification criteria below

## QUADRUPLE

Quadruple ID: {QUADRUPLE_ID}
Shared surface marker: "{SHARED_MARKER}"

Passage P1 (passage_id: {QUADRUPLE_ID}-P1, domain: {P1_DOMAIN}{P1_POS_SUFFIX}):
{P1_TEXT}

Passage P2 (passage_id: {QUADRUPLE_ID}-P2, domain: {P2_DOMAIN}{P2_POS_SUFFIX}):
{P2_TEXT}

Passage P3 (passage_id: {QUADRUPLE_ID}-P3, domain: {P3_DOMAIN}{P3_POS_SUFFIX}):
{P3_TEXT}

Passage P4 (passage_id: {QUADRUPLE_ID}-P4, domain: {P4_DOMAIN}{P4_POS_SUFFIX}):
{P4_TEXT}

## BANK-LEVEL METADATA (for attestation 5c)

This quadruple is part of a {N_QUADRUPLES}-quadruple pilot bank. Across all {N_QUADRUPLES} quadruples in this bank, the trans_discourse passages cover the following position types: {TRANS_POSITION_TYPES_LIST}.

## VERIFICATION ATTESTATIONS

You attest to the following 10 statements about this quadruple. Each is a binary attestation (true / false; `external_qualification` is a short descriptor string).

1. **attests_5a_shared_surface_marker**: The shared surface marker "{SHARED_MARKER}" appears in all 4 passages of this quadruple as a standalone word or as an immediate morphological variant (e.g., "harm" matches "harms"/"harmed"; "safe" matches "safety"). (true/false)

2. **attests_5b_no_anticipated_feature_profile**: Based on the passages alone, there is no evidence that these passages were selected to produce a predetermined feature-schema profile outcome. (true/false)

3. **attests_5c_position_types_span_three_or_more**: Across the bank as a whole (per the metadata above), the trans_discourse passages span at least 3 of the 5 position types (professional_consensus, cautious_clinical_review, model_legislation_findings, advocacy_legal, skeptical_academic). (true/false)

4. **attests_6_elision_for_length_only**: Any `[...]` elisions present in these passages appear to be for length compliance only, not for content selection or rhetorical shaping. (true/false)

5. **external_qualification**: Short string describing your qualification as an external reviewer (e.g., "no_prior_GDA_participation").

6. **no_co_authorship**: You have no co-authorship relationship with the study director. (true/false)

7. **blinded_to_hypothesis_structure**: You have not been shown the study's hypothesis structure. (true/false)

8. **received_redacted_passages_only**: You received only the 4 redacted passages and the metadata listed above, with no additional source information. (true/false)

9. **selection_reasoning_archived_pre_schema**: Your selection_rationale field below is produced before any feature-schema characterization. (true/false)

10. **no_evidence_of_curation_for_outcomes**: You see no evidence that the passages were curated to produce a specific predetermined comparison outcome on a feature schema you have not been shown. (true/false)

## MECHANICAL BIAS CHECK (per §4)

For each of the 6 unique passage pairs in fixed order [P1-P2, P1-P3, P1-P4, P2-P3, P2-P4, P3-P4], give your pre-analytic intuitive estimate of structural similarity on a 0–7 integer scale, where:
- 0 = nearly identical structure (same rhetorical architecture, comparable evidentiary stance)
- 7 = maximally different structure (entirely different rhetorical architecture)

These are intuitive guesses produced BEFORE any feature-schema characterization. You are not being shown the feature schema. Output as a list of exactly 6 integers in the order above.

## SELECTION RATIONALE

In 100–200 words, describe what structural or rhetorical features (beyond the shared surface marker, which you and the other reviewer trivially share) you think unite these specific passages. What appears to be the principle behind grouping them together?

## OUTPUT SCHEMA

Return exactly this JSON structure. Return ONLY the JSON, no preamble, no markdown fences:

{
  "pilot_quadruple_id": "{QUADRUPLE_ID}",
  "attests_5a_shared_surface_marker": true,
  "attests_5b_no_anticipated_feature_profile": true,
  "attests_5c_position_types_span_three_or_more": true,
  "attests_6_elision_for_length_only": true,
  "external_qualification": "no_prior_GDA_participation",
  "no_co_authorship": true,
  "blinded_to_hypothesis_structure": true,
  "received_redacted_passages_only": true,
  "selection_reasoning_archived_pre_schema": true,
  "no_evidence_of_curation_for_outcomes": true,
  "mechanical_bias_check_guess_distance": [3, 4, 5, 4, 5, 4],
  "selection_rationale": ""
}"""


# ========================================================================
# ADJUDICATOR PROMPT TEMPLATE
# ========================================================================

ADJUDICATOR_PROMPT_TEMPLATE = """You are an adjudicator for a research methodology study. Two independent reviewers have each been shown the same quadruple of 4 redacted text passages and asked to describe what structural or rhetorical features unite the passages. Your job is to score the substantive overlap between their two rationales.

You are NOT scoring whether either rationale is "correct" or "well-written". You are scoring whether the two reviewers, working independently, identified substantively overlapping structural features.

## QUADRUPLE METADATA

Quadruple ID: {QUADRUPLE_ID}
Shared surface marker (visible to both reviewers): "{SHARED_MARKER}"

## REVIEWER 1 RATIONALE

{R1_RATIONALE}

## REVIEWER 2 RATIONALE

{R2_RATIONALE}

## SCORING TASK

Score the substantive overlap on a 0.0–1.0 continuous scale:

- 1.0 = The two rationales identify essentially the same set of structural features uniting the passages, even if phrased differently.
- 0.7–0.9 = Substantially overlapping features, with some differences in emphasis or coverage.
- 0.4–0.6 = Some shared features but also divergence on key structural claims.
- 0.1–0.3 = Mostly different features, with only minor incidental overlap.
- 0.0 = Entirely different features with no meaningful overlap.

DO NOT count the shared surface marker itself as a structural feature for the purpose of scoring overlap — both reviewers were given that and so trivially share it.

DO count: overlap on rhetorical architecture (e.g., both describe "appeal to consensus", "uncertainty management", "burden-of-proof framing"), evidentiary stance, institutional posture, action posture, audience-orientation, or any other deeper rhetorical-structural feature the reviewers independently identified.

## OUTPUT SCHEMA

Return exactly this JSON structure. Return ONLY the JSON, no preamble, no markdown fences:

{
  "pilot_quadruple_id": "{QUADRUPLE_ID}",
  "overlap_score": 0.0,
  "shared_features_identified": ["feature1", "feature2"],
  "divergent_features": ["only_R1_feature1", "only_R2_feature1"],
  "scoring_reasoning": "brief 2-3 sentence explanation of the score"
}"""


# ========================================================================
# UTILITIES
# ========================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bank_jsonl(text: str) -> list:
    bank = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        bank.append(json.loads(line))
    return bank


def extract_json(text: str) -> dict:
    """Extract first valid JSON object from a model response. Forgiving of
    markdown fences and preamble; does not synthesize."""
    text = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            if start is None:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    depth = 0
    raise ValueError(f"Could not extract valid JSON from response: {text[:300]!r}...")


# ========================================================================
# SCHEMA VALIDATION
# ========================================================================

ALLOWED_DOMAINS = {
    "tobacco", "climate", "antivax", "public-health-vax",
    "ai_welfare", "trans_discourse",
}
SETTLED_DOMAINS = {"tobacco", "climate", "antivax", "public-health-vax"}
ALLOWED_TRANS_POSITIONS = {
    "professional_consensus", "cautious_clinical_review",
    "model_legislation_findings", "advocacy_legal", "skeptical_academic",
}
WORD_MIN, WORD_MAX = 150, 250


def marker_present(text: str, marker: str) -> bool:
    pattern = r"\b" + re.escape(marker.lower()) + r"\w*"
    return bool(re.search(pattern, text.lower()))


def validate_bank(bank: list) -> tuple:
    errors, warnings = [], []
    if not bank:
        errors.append("bank is empty")
        return errors, warnings

    for q in bank:
        qid = q.get("pilot_quadruple_id", "<missing>")
        if not all(k in q for k in ["pilot_quadruple_id", "shared_surface_marker", "passages"]):
            errors.append(f"[{qid}] missing required field(s)")
            continue
        passages = q["passages"]
        if len(passages) != 4:
            errors.append(f"[{qid}] expected 4 passages, got {len(passages)}")
            continue
        marker = q["shared_surface_marker"]

        domains = [p.get("domain") for p in passages]
        settled = sum(1 for d in domains if d in SETTLED_DOMAINS)
        ai = sum(1 for d in domains if d == "ai_welfare")
        trans = sum(1 for d in domains if d == "trans_discourse")
        if settled != 2:
            errors.append(f"[{qid}] expected 2 settled-domain passages, got {settled}")
        if ai != 1:
            errors.append(f"[{qid}] expected 1 ai_welfare passage, got {ai}")
        if trans != 1:
            errors.append(f"[{qid}] expected 1 trans_discourse passage, got {trans}")

        for p in passages:
            pid = p.get("passage_id", "<missing>")
            if p.get("domain") not in ALLOWED_DOMAINS:
                errors.append(f"  [{pid}] invalid domain")
            pos = p.get("position_type")
            if p.get("domain") == "trans_discourse":
                if pos not in ALLOWED_TRANS_POSITIONS:
                    errors.append(f"  [{pid}] invalid trans position_type '{pos}'")
            text = p.get("text", "")
            wc = len(text.split())
            if wc < WORD_MIN or wc > WORD_MAX:
                errors.append(f"  [{pid}] word count {wc} outside [{WORD_MIN}, {WORD_MAX}]")
            if p.get("source_attribution") != "REDACTED":
                errors.append(f"  [{pid}] source_attribution must be 'REDACTED'")
            if not marker_present(text, marker):
                errors.append(f"  [{pid}] shared marker '{marker}' not found")

    trans_positions = set()
    for q in bank:
        for p in q.get("passages", []):
            if p.get("domain") == "trans_discourse":
                if p.get("position_type"):
                    trans_positions.add(p["position_type"])
    if len(trans_positions) < 3:
        errors.append(f"BANK: trans position types must span >=3, found only {sorted(trans_positions)}")

    return errors, warnings


def get_bank_trans_position_types(bank: list) -> list:
    positions = set()
    for q in bank:
        for p in q.get("passages", []):
            if p.get("domain") == "trans_discourse" and p.get("position_type"):
                positions.add(p["position_type"])
    return sorted(positions)


def verify_lab_class_separation(r1: dict, r2: dict) -> tuple:
    """Per doc 23 §3.6 and doc 19 §3.5: return (separated_boolean, n_differing)
    where separated_boolean is True iff the two lab_class objects differ on
    at least 3 of the 5 properties."""
    props = [
        "primary_training_language",
        "rlhf_philosophy",
        "training_data_overlap_with_us_frontier",
        "alignment_taste_documented",
        "architecture_and_origin",
    ]
    differing = [p for p in props if r1["lab_class"].get(p) != r2["lab_class"].get(p)]
    return (len(differing) >= 3, len(differing), differing)


# ========================================================================
# PROMPT ASSEMBLY
# ========================================================================

def build_reviewer_prompt(template: str, q: dict, bank_metadata: dict) -> str:
    out = template
    out = out.replace("{QUADRUPLE_ID}", q["pilot_quadruple_id"])
    out = out.replace("{SHARED_MARKER}", q["shared_surface_marker"])
    out = out.replace("{N_QUADRUPLES}", str(bank_metadata["n_quadruples"]))
    out = out.replace("{TRANS_POSITION_TYPES_LIST}",
                      ", ".join(bank_metadata["trans_position_types"]))
    for i, p in enumerate(q["passages"], start=1):
        pos = p.get("position_type")
        pos_suffix = f", position_type: {pos}" if pos else ""
        out = out.replace(f"{{P{i}_DOMAIN}}", p["domain"])
        out = out.replace(f"{{P{i}_POS_SUFFIX}}", pos_suffix)
        out = out.replace(f"{{P{i}_TEXT}}", p["text"])
    return out


def build_adjudicator_prompt(template: str, q: dict, r1: str, r2: str) -> str:
    out = template
    out = out.replace("{QUADRUPLE_ID}", q["pilot_quadruple_id"])
    out = out.replace("{SHARED_MARKER}", q["shared_surface_marker"])
    out = out.replace("{R1_RATIONALE}", r1)
    out = out.replace("{R2_RATIONALE}", r2)
    return out


# ========================================================================
# OPENROUTER CALL
# ========================================================================

def call_openrouter(model_id: str, prompt: str, api_key: str) -> dict:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        **SAMPLING,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/phase4d-pilot",
        "X-Title": "Phase 4D External Reviewer Pilot",
    }
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = now_iso()
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            t1 = now_iso()
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {
                "raw_text": content,
                "request_started_at": t0,
                "request_completed_at": t1,
                "model_id": model_id,
                "sampling_params": SAMPLING,
                "openrouter_response_id": data.get("id"),
                "usage": data.get("usage"),
                "attempt": attempt,
            }
        except Exception as e:
            last_err = e
            print(f"    [attempt {attempt}/{MAX_RETRIES}] {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"OpenRouter call failed after {MAX_RETRIES} attempts: {last_err}")


# ========================================================================
# METRICS (per doc 24 §5)
# ========================================================================

def compute_feature_schema_attestation_match_rate(r1_atts: list, r2_atts: list) -> dict:
    """Doc 24 §5 metric 1: fraction of identical responses on the 10 binary
    attestation fields (quadruple-level per doc 23 §3.6).

    For n quadruples × 10 binary fields per reviewer = 10n paired attestations.
    """
    total = 0
    matched = 0
    per_field = {}
    per_quadruple = []

    for r1, r2 in zip(r1_atts, r2_atts):
        q_matched = 0
        q_total = 0
        for field in QUAD_LEVEL_BINARY_FIELDS:
            v1 = r1.get(field) if isinstance(r1, dict) else None
            v2 = r2.get(field) if isinstance(r2, dict) else None
            agree = (v1 == v2)
            per_field.setdefault(field, {"matched": 0, "total": 0})
            per_field[field]["total"] += 1
            if agree:
                per_field[field]["matched"] += 1
            total += 1
            q_total += 1
            if agree:
                matched += 1
                q_matched += 1
        qid = r1.get("pilot_quadruple_id") if isinstance(r1, dict) else None
        per_quadruple.append({
            "pilot_quadruple_id": qid,
            "matched": q_matched,
            "total": q_total,
            "rate": q_matched / q_total if q_total > 0 else 0.0,
        })

    rate = matched / total if total > 0 else 0.0
    return {
        "rate": rate,
        "matched": matched,
        "total": total,
        "per_field": per_field,
        "per_quadruple": per_quadruple,
    }


def compute_mechanical_bias_check_correlation(r1_atts: list, r2_atts: list) -> dict:
    """Doc 24 §5 metric 2: Pearson r on Hamming-distance guesses.

    Per doc 23 §3.6: mechanical_bias_check_guess_distance is a list of 6
    integers in fixed order [P1-P2, P1-P3, P1-P4, P2-P3, P2-P4, P3-P4].

    INFORMATIONAL ONLY — does not enter the decision rule (doc 24 §5 line 79).
    """
    r1_scores, r2_scores, pair_rows = [], [], []
    for r1, r2 in zip(r1_atts, r2_atts):
        qid = r1.get("pilot_quadruple_id", "?") if isinstance(r1, dict) else "?"
        r1_list = r1.get("mechanical_bias_check_guess_distance") if isinstance(r1, dict) else None
        r2_list = r2.get("mechanical_bias_check_guess_distance") if isinstance(r2, dict) else None
        if not isinstance(r1_list, list) or not isinstance(r2_list, list):
            continue
        for i, pair in enumerate(PAIRS_ORDER):
            if i >= len(r1_list) or i >= len(r2_list):
                continue
            try:
                v1 = float(r1_list[i])
                v2 = float(r2_list[i])
            except (TypeError, ValueError):
                continue
            r1_scores.append(v1)
            r2_scores.append(v2)
            pair_rows.append({"pilot_quadruple_id": qid, "pair": pair, "r1": v1, "r2": v2})

    if len(r1_scores) < 2:
        return {"r": None, "n_pairs": len(r1_scores), "pair_data": pair_rows,
                "note": "insufficient paired observations"}
    a = np.array(r1_scores)
    b = np.array(r2_scores)
    if a.std() == 0 or b.std() == 0:
        return {"r": None, "n_pairs": len(r1_scores), "pair_data": pair_rows,
                "note": "zero variance in one or both series"}
    r = float(np.corrcoef(a, b)[0, 1])
    return {"r": r, "n_pairs": len(r1_scores), "pair_data": pair_rows}


def compute_selection_rationale_overlap(adj_results: list) -> dict:
    """Doc 24 §5 metric 3: mean adjudicator overlap score across quadruples."""
    scores = [a["overlap_score"] for a in adj_results if a.get("overlap_score") is not None]
    if not scores:
        return {"mean": None, "n": 0, "scores": []}
    return {"mean": float(np.mean(scores)), "n": len(scores), "scores": scores}


# ========================================================================
# REVIEWER RECORD ASSEMBLY (per doc 23 §3.6)
# ========================================================================

def assemble_reviewer_record(reviewer_cfg: dict, parsed_attestation: dict,
                             call_resp: dict, prompt: str) -> dict:
    """Build a single §3.6-compliant reviewer record from a parsed attestation
    plus the OpenRouter call metadata."""
    record = {
        "reviewer_id": reviewer_cfg["reviewer_id"],
        "reviewer_type": reviewer_cfg["reviewer_type"],
        "lab_class": reviewer_cfg["lab_class"],
    }
    # Carry the 10 binary attestation fields through (with defaults if missing)
    for field in QUAD_LEVEL_BINARY_FIELDS:
        record[field] = parsed_attestation.get(field) if isinstance(parsed_attestation, dict) else None
    # Mechanical bias check and rationale
    record["mechanical_bias_check_guess_distance"] = (
        parsed_attestation.get("mechanical_bias_check_guess_distance")
        if isinstance(parsed_attestation, dict) else None
    )
    record["selection_rationale"] = (
        parsed_attestation.get("selection_rationale", "")
        if isinstance(parsed_attestation, dict) else ""
    )
    # Carry through any parse error info for transparency
    if isinstance(parsed_attestation, dict) and "_parse_error" in parsed_attestation:
        record["_parse_error"] = parsed_attestation["_parse_error"]
        record["_raw_text"] = parsed_attestation.get("_raw_text", "")
    # Reproducibility archive per doc 23 §3.6
    record["reproducibility_archive"] = {
        "reviewer_prompt": prompt,
        "model_identifier": reviewer_cfg["model_id"],
        "model_version_hash": f"openrouter-response-id:{call_resp.get('openrouter_response_id')}",
        "sampling_parameters": SAMPLING,
        "raw_output": call_resp["raw_text"],
        "attestation_timestamp": call_resp["request_completed_at"],
    }
    record["signature_date"] = call_resp["request_completed_at"]
    record["signature_method"] = "model-id-and-version-hash"
    return record


# ========================================================================
# MAIN
# ========================================================================

def main():
    print("=" * 70)
    print("Phase 4D External Reviewer Pilot - Consolidated Runner v1.2")
    print("(Aligned with doc 19, doc 23 §3.6, doc 24)")
    print("=" * 70)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("\nERROR: OPENROUTER_API_KEY environment variable not set.")
        print("       Set it before running:")
        print("           export OPENROUTER_API_KEY=sk-or-...")
        print("       Or in a notebook cell BEFORE running this script:")
        print("           import os; os.environ['OPENROUTER_API_KEY'] = 'sk-or-...'")
        sys.exit(2)

    # 1. Write bank to disk (committable artifact)
    print(f"\nWriting bank to {BANK_OUT_PATH} (commit this to GitHub)")
    with BANK_OUT_PATH.open("w", encoding="utf-8") as f:
        f.write(BANK_JSONL.strip() + "\n")

    # 2. Parse and validate bank
    bank = parse_bank_jsonl(BANK_JSONL)
    print(f"\nValidating {len(bank)} quadruples...")
    errors, warnings = validate_bank(bank)
    if warnings:
        for w in warnings:
            print(f"  ! {w}")
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  X {e}")
        print("\nFAILED VALIDATION. Aborting pilot run.")
        sys.exit(1)

    bank_metadata = {
        "n_quadruples": len(bank),
        "trans_position_types": get_bank_trans_position_types(bank),
    }
    print(f"  VALID. Trans position types covered ({len(bank_metadata['trans_position_types'])}/5): "
          f"{bank_metadata['trans_position_types']}")

    # 3. Verify lab_class separation per doc 23 §3.6 / doc 19 §3.5
    separated, n_diff, diff_props = verify_lab_class_separation(
        REVIEWERS[0], REVIEWERS[1]
    )
    print(f"\nLab_class separation check (doc 19 §3.5):")
    print(f"  Properties differing: {n_diff}/5 (required: >=3)")
    for p in diff_props:
        print(f"    - {p}")
    if not separated:
        print(f"  FAIL: reviewers do not meet non-overlapping-ancestry requirement.")
        sys.exit(1)
    print(f"  PASS: reviewers exceed the >=3 separation requirement.")

    print(f"\nReviewers:")
    for r in REVIEWERS:
        print(f"  {r['reviewer_id']} ({r['reviewer_label']}): {r['model_id']}")
    print(f"Adjudicator: {ADJUDICATOR['label']} ({ADJUDICATOR['model_id']})")
    print(f"Threshold for binding metrics: {THRESHOLD}")
    print(f"\nDeviation from spec: n={len(bank)} (pre-reg n=5); "
          f"see methods/limitations.")

    # 4. Run reviewers
    print("\n" + "-" * 70)
    print("Phase 1: Running reviewers")
    print("-" * 70)
    archive = []
    attestations = {r["reviewer_id"]: [] for r in REVIEWERS}
    reviewer_records = {r["reviewer_id"]: [] for r in REVIEWERS}

    n_calls = len(bank) * len(REVIEWERS)
    call_idx = 0
    for q in bank:
        for reviewer in REVIEWERS:
            call_idx += 1
            qid = q["pilot_quadruple_id"]
            rid = reviewer["reviewer_id"]
            print(f"  [{call_idx}/{n_calls}] {qid} | {rid} ({reviewer['reviewer_label']})")
            prompt = build_reviewer_prompt(
                REVIEWER_PROMPT_TEMPLATE, q, bank_metadata
            )
            try:
                resp = call_openrouter(reviewer["model_id"], prompt, api_key)
            except Exception as e:
                print(f"    FATAL: {e}")
                sys.exit(1)
            try:
                attestation = extract_json(resp["raw_text"])
                # Force the pilot_quadruple_id to match what we sent (in case
                # the reviewer returns a malformed value)
                attestation["pilot_quadruple_id"] = qid
            except Exception as e:
                print(f"    JSON parse failed: {e}")
                attestation = {
                    "pilot_quadruple_id": qid,
                    "_parse_error": str(e),
                    "_raw_text": resp["raw_text"],
                }
            attestations[rid].append(attestation)
            reviewer_records[rid].append(
                assemble_reviewer_record(reviewer, attestation, resp, prompt)
            )
            archive.append({
                "step": "reviewer_call",
                "pilot_quadruple_id": qid,
                "reviewer_id": rid,
                "reviewer_label": reviewer["reviewer_label"],
                "model_id": reviewer["model_id"],
                "response_meta": {
                    "request_started_at": resp["request_started_at"],
                    "request_completed_at": resp["request_completed_at"],
                    "openrouter_response_id": resp["openrouter_response_id"],
                    "usage": resp["usage"],
                    "attempt": resp["attempt"],
                },
            })

    # 5. Run adjudicator
    print("\n" + "-" * 70)
    print("Phase 2: Running rationale-overlap adjudicator")
    print("-" * 70)
    adj_results = []
    r1_atts = attestations[REVIEWERS[0]["reviewer_id"]]
    r2_atts = attestations[REVIEWERS[1]["reviewer_id"]]
    for i, q in enumerate(bank):
        qid = q["pilot_quadruple_id"]
        r1_rat = r1_atts[i].get("selection_rationale", "") if isinstance(r1_atts[i], dict) else ""
        r2_rat = r2_atts[i].get("selection_rationale", "") if isinstance(r2_atts[i], dict) else ""
        if not r1_rat or not r2_rat:
            print(f"  [SKIP] {qid}: missing rationale from one or both reviewers")
            adj_results.append({"pilot_quadruple_id": qid, "overlap_score": None,
                                "_skipped_reason": "missing_rationale"})
            continue
        print(f"  {qid}")
        prompt = build_adjudicator_prompt(
            ADJUDICATOR_PROMPT_TEMPLATE, q, r1_rat, r2_rat
        )
        try:
            resp = call_openrouter(ADJUDICATOR["model_id"], prompt, api_key)
        except Exception as e:
            print(f"    FATAL: {e}")
            sys.exit(1)
        try:
            adj_obj = extract_json(resp["raw_text"])
            adj_obj["pilot_quadruple_id"] = qid
        except Exception as e:
            print(f"    JSON parse failed: {e}")
            adj_obj = {"pilot_quadruple_id": qid, "overlap_score": None,
                       "_parse_error": str(e), "_raw_text": resp["raw_text"]}
        adj_results.append(adj_obj)
        archive.append({
            "step": "adjudicator_call",
            "pilot_quadruple_id": qid,
            "adjudicator_id": ADJUDICATOR["model_id"],
            "response_meta": {
                "request_started_at": resp["request_started_at"],
                "request_completed_at": resp["request_completed_at"],
                "openrouter_response_id": resp["openrouter_response_id"],
                "usage": resp["usage"],
            },
        })

    # 6. Compute metrics
    print("\n" + "-" * 70)
    print("Phase 3: Computing convergence metrics (per doc 24 §5)")
    print("-" * 70)
    match_metric = compute_feature_schema_attestation_match_rate(r1_atts, r2_atts)
    pearson_metric = compute_mechanical_bias_check_correlation(r1_atts, r2_atts)
    overlap_metric = compute_selection_rationale_overlap(adj_results)

    print(f"\n  BINDING METRICS (enter the decision rule):")
    print(f"    feature_schema_attestation_match_rate:    {match_metric['rate']:.3f} "
          f"({match_metric['matched']}/{match_metric['total']})")
    if overlap_metric["mean"] is not None:
        print(f"    selection_rationale_substantive_overlap:  {overlap_metric['mean']:.3f} "
              f"(n={overlap_metric['n']} adjudications)")
    else:
        print(f"    selection_rationale_substantive_overlap:  COULD NOT COMPUTE")
    print(f"\n  INFORMATIONAL (NOT in decision rule, per doc 24 §5):")
    if pearson_metric["r"] is not None:
        print(f"    mechanical_bias_check_guess_distance_correlation: {pearson_metric['r']:.3f} "
              f"(n={pearson_metric['n_pairs']} paired guesses)")
    else:
        print(f"    mechanical_bias_check_guess_distance_correlation: COULD NOT COMPUTE "
              f"({pearson_metric.get('note', 'unknown')})")

    # 7. Decision rule per doc 24 §5 (INVERTED logic — high convergence is the failure mode)
    print("\n" + "-" * 70)
    print("Phase 4: Decision rule (doc 24 §5)")
    print("-" * 70)

    match_above = match_metric["rate"] > THRESHOLD
    overlap_above = (overlap_metric["mean"] is not None
                     and overlap_metric["mean"] > THRESHOLD)

    if match_above and overlap_above:
        decision_rule_outcome = "FAILED_INVOKE_A_B_C"
        decision_msg = (
            "Both binding metrics exceed 0.80:\n"
            f"  - feature_schema_attestation_match_rate = {match_metric['rate']:.3f} > {THRESHOLD}\n"
            f"  - selection_rationale_substantive_overlap = {overlap_metric['mean']:.3f} > {THRESHOLD}\n\n"
            "Per doc 24 §5: the empirical premise for the LLM-only argument has "
            "NOT generalized to this one-shot reviewer task. The two reviewers "
            "converge more than the design quorum did under sustained scrutiny. "
            "Director must respond with one of (per §5):\n\n"
            "  (a) Add a third LLM reviewer differing from R01 and R02 on >=3 "
            "of the 5 lab_class properties. Re-run pilot with all three "
            "reviewers; re-evaluate decision rule against larger reviewer "
            "set.\n      -> commit outcome as: failed_added_third\n\n"
            "  (b) Add a sensitivity-check human reviewer (academic philosophy-"
            "of-science / methodology pool per §3.5). Human attestations are "
            "pooled with LLMs on equal footing, not substituted. Bank "
            "construction proceeds with three reviewers.\n      -> commit "
            "outcome as: failed_added_human\n\n"
            "  (c) Document the pilot result as a limitation in the methodology "
            "paper and proceed with LLM-only composition under documented "
            "reduced confidence. Permitted only if director judges a third "
            "reviewer would not materially change the outcome.\n      -> commit "
            "outcome as: failed_documented_limitation\n"
        )
    else:
        decision_rule_outcome = "passed"
        reasons = []
        if not match_above:
            reasons.append(
                f"feature_schema_attestation_match_rate = "
                f"{match_metric['rate']:.3f} <= {THRESHOLD}"
            )
        if not overlap_above:
            v = overlap_metric["mean"]
            reasons.append(
                f"selection_rationale_substantive_overlap = "
                f"{v if v is None else round(v, 3)} <= {THRESHOLD}"
            )
        decision_msg = (
            "At least one binding metric is at or below 0.80:\n  - "
            + "\n  - ".join(reasons)
            + "\n\nPer doc 24 §5: the empirical premise for the LLM-only "
              "argument HAS generalized to this one-shot reviewer task. "
              "The two reviewers retain sufficient differentiation. "
              "LLM-only composition proceeds to full bank construction.\n"
              "      -> commit outcome as: passed"
        )

    print(f"\n  *** decision_rule_outcome: {decision_rule_outcome} ***\n")
    for line in decision_msg.split("\n"):
        print(f"  {line}")

    # 8. Write output per doc 23 §5.5 + doc 24 §6 format
    print("\n" + "-" * 70)
    print("Phase 5: Writing pilot artifact (doc 23 §5.5 + doc 24 §6 format)")
    print("-" * 70)

    # Per-quadruple records
    per_quadruple_records = []
    for i, q in enumerate(bank):
        qid = q["pilot_quadruple_id"]
        # Per-quadruple sub-metrics
        quad_match = compute_feature_schema_attestation_match_rate(
            [r1_atts[i]] if i < len(r1_atts) else [],
            [r2_atts[i]] if i < len(r2_atts) else [],
        )
        quad_pearson = compute_mechanical_bias_check_correlation(
            [r1_atts[i]] if i < len(r1_atts) else [],
            [r2_atts[i]] if i < len(r2_atts) else [],
        )
        quad_adj = adj_results[i] if i < len(adj_results) else {}
        quad_overlap = quad_adj.get("overlap_score")

        # Assemble full reviewer records per §3.6
        per_quadruple_records.append({
            "pilot_quadruple_id": qid,
            "passages": q["passages"],
            "shared_surface_marker": q["shared_surface_marker"],
            "reviewer_attestations": [
                reviewer_records[REVIEWERS[0]["reviewer_id"]][i],
                reviewer_records[REVIEWERS[1]["reviewer_id"]][i],
            ],
            "convergence_metrics": {
                "feature_schema_attestation_match_rate": quad_match["rate"],
                "mechanical_bias_check_guess_distance_correlation": quad_pearson["r"],
                "selection_rationale_substantive_overlap": quad_overlap,
            },
            "adjudication": quad_adj,
            "lab_class_separation_verified": separated,
        })

    # Summary record per doc 24 §6
    summary_record = {
        "pilot_summary": True,
        "n_quadruples": len(bank),
        "n_reviewers": len(REVIEWERS),
        "aggregate_match_rate": match_metric["rate"],
        "aggregate_correlation": pearson_metric["r"],
        "aggregate_overlap": overlap_metric["mean"],
        "decision_rule_outcome": decision_rule_outcome,
        "rationale": decision_msg,
        "reviewers": REVIEWERS,
        "adjudicator": ADJUDICATOR,
        "threshold": THRESHOLD,
        "sampling_parameters": SAMPLING,
        "lab_class_separation_verified": separated,
        "lab_class_separation_n_differing": n_diff,
        "pre_reg_deviation_note": (
            "Pilot run at n=4 instead of pre-registered n=5; "
            "skeptical_academic position type omitted due to absence of "
            "open-access sources meeting verbatim/length constraints. "
            "Documented in paper methods/limitations per the no-more-"
            "amendments decision. Trans positions covered: 4 of 5."
        ),
        "completed_at_iso": now_iso(),
        "metrics_detail": {
            "feature_schema_attestation_match_rate": match_metric,
            "mechanical_bias_check_guess_distance_correlation": pearson_metric,
            "selection_rationale_substantive_overlap": overlap_metric,
        },
        "reproducibility_archive_call_log": archive,
    }

    with PILOT_OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in per_quadruple_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

    print(f"  Wrote: {PILOT_OUT_PATH}")
    print(f"\n  COMMIT TO GITHUB (regardless of outcome, per pre-reg):")
    print(f"    - {BANK_OUT_PATH}")
    print(f"    - {PILOT_OUT_PATH}")
    print(f"    - this script")

    print("\n" + "=" * 70)
    print(f"  decision_rule_outcome: {decision_rule_outcome}")
    print("=" * 70)
    sys.exit(0 if decision_rule_outcome == "passed" else 1)


if __name__ == "__main__":
    main()
