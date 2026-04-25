# ZerveXHackerEarth (Clone) - Canvas DAG

- Blocks: **67**
- Edges: **78**
- Source: `canvas.yaml`

```mermaid
flowchart LR
  classDef code fill:#e6f2ff,stroke:#3b82f6,color:#0b2545;
  classDef note fill:#fff7d6,stroke:#d4a017,color:#4a3a00;
  classDef llm  fill:#f3e8ff,stroke:#8b5cf6,color:#2e1065;
  subgraph ingest["Ingestion & EDA"]
    direction TB
    nc4eb7a45["Example Dataset"]
    n44911432["Data Exploration"]
    n858cac69["Statistical Summaries"]
    n588ecfb2["02_statistical_profiling"]
    nad734095["User Activity Analysis"]
    nb0f6ada0["Temporal Patterns"]
    n95d4565f["User Type & Anomaly Detection"]
    nf9026078["Additional Exploratory Visualizations"]
    n2cc6ed75(["01_data_loading_and_overview"])
    n1aafa816["load_and_inspect_dataset"]
    n031610c4["date_range_and_temporal_coverage"]
    nb16a055c["data_quality_checks"]
    n46a3584c["Dataset Field-Level Description"]
  end
  subgraph semantics["Event Semantics"]
    direction TB
    nd457c3be["Event Taxonomy & Categorization"]
    ne22b5687["Workflow Stage Mapping"]
    n2a61d9e1["Hierarchical Event Visualization"]
  end
  subgraph segment["Segmentation"]
    direction TB
    n48382513["Engagement Segmentation"]
    n6cabbeab["Feature Adoption Evolution"]
    n210add96["Workflow Pattern Segmentation"]
    n4aafa92c["Temporal & Monetization Segmentation"]
    n4d3ec133["Interactive Visualizations & Segment Export"]
    n10098bbf["Feature Adoption Trajectories"]
    n8a6e2af1["Lifecycle Stage Definition"]
    n8372a34c["Growth Trajectory Classification"]
  end
  subgraph behavior["Behavioral Features"]
    direction TB
    n421e0ec3["Comprehensive Feature Engineering"]
    n1bfb2fb3["Session Pattern Analysis"]
    na95f5629["Workflow Sequence Patterns"]
    ne0470225["Collaboration Signature & Final Matrix"]
    n2c9d29bd["Isolation Forest Anomaly Detection"]
    n56a67c43["Engagement Forecast per Segment (Copy)"]
    n5360779a["Engagement Momentum Tracking"]
    nbb4bb303["Engagement Forecast per Segment"]
    nfe9a5757["Cohort Behavioral DNA — Feature Engineering"]
  end
  subgraph modeling["Modeling / Churn / LTV"]
    direction TB
    n4fe8f20e["Primary Success Metrics"]
    n6fddc4f7["Composite Success Score & Labeling"]
    n4dc5810c["Validation & Business Alignment"]
    n4a843183["01_data_prep_train_val_test_split"]
    n6f1f3592["02_base_models_ensemble"]
    n3e0b74e7["Survival Analysis Data Preparation"]
    naea140e4["Kaplan-Meier Survival Curves by Segment"]
    n35e752f4["Churn Risk Scoring & Time-Based Predictions"]
    n5215a49f["LTV Prediction & Unit Economics"]
    n6a01f22c["Behavioral Economics Scoring"]
    n8986b069["Churn Early Warning System"]
    n0e209692["Churn Early Warning — Ranked Action Table"]
  end
  subgraph graph["Graph / GNN"]
    direction TB
    n8c234a81["Collaboration Network & Centrality Analysis"]
    na5bb52f8["Community Detection & Success Correlation"]
    ncac932c5["GNN Social Influence Graph Construction"]
    na1e3894b["GraphSAGE Training & Social Influence Embeddings"]
    nc5297c1b["Hybrid GNN Churn Model & Community Analysis"]
  end
  subgraph report["Reporting & Export"]
    direction TB
    ne4e0c430["Advanced Analysis Synthesis"]
    n4865d3d4(["Comprehensive Analysis Report"])
    n88d4c03d["Integrated Dashboard Synthesis"]
    nde6eadd5["Causal Impact & Attribution Analysis"]
    nde48e1fc["SHAP Explainability Analysis"]
    nae21e716["Export Report as Markdown File"]
    n6e1621cd["Weekly Delta Metrics Computation"]
    n3aaed87c{{"Weekly Insights Executive Briefing"}}
    n45dd0f5b["Save Report to File"]
    n7ba033e3(["key_findings_summary"])
    n79cecf00["Social Media Post Drafts"]
    nbf4d7573(["Project README"])
    n043b0f20(["Quality Assurance Checklist"])
    n3b852862["Executive Summary Report Generation"]
    nfe7a6c1f["User Intelligence Export"]
    ncde7f66a["Git Repository Setup"]
  end
  subgraph other["Other"]
    direction TB
    n5eacdfde["Comprehensive User Analysis Findings"]
  end
  class n031610c4,n0e209692,n10098bbf,n1aafa816,n1bfb2fb3,n210add96,n2a61d9e1,n2c9d29bd,n35e752f4,n3b852862,n3e0b74e7,n421e0ec3,n44911432,n45dd0f5b,n46a3584c,n48382513,n4a843183,n4aafa92c,n4d3ec133,n4dc5810c,n4fe8f20e,n5215a49f,n5360779a,n56a67c43,n588ecfb2,n5eacdfde,n6a01f22c,n6cabbeab,n6e1621cd,n6f1f3592,n6fddc4f7,n79cecf00,n8372a34c,n858cac69,n88d4c03d,n8986b069,n8a6e2af1,n8c234a81,n95d4565f,na1e3894b,na5bb52f8,na95f5629,nad734095,nae21e716,naea140e4,nb0f6ada0,nb16a055c,nbb4bb303,nc4eb7a45,nc5297c1b,ncac932c5,ncde7f66a,nd457c3be,nde48e1fc,nde6eadd5,ne0470225,ne22b5687,ne4e0c430,nf9026078,nfe7a6c1f,nfe9a5757 code;
  class n043b0f20,n2cc6ed75,n4865d3d4,n7ba033e3,nbf4d7573 note;
  class n3aaed87c llm;
  ne4e0c430 --> n6a01f22c
  n4d3ec133 --> n5215a49f
  nc4eb7a45 --> n588ecfb2
  n44911432 --> n95d4565f
  n1aafa816 --> n031610c4
  n5360779a --> nbb4bb303
  n2c9d29bd --> n88d4c03d
  n44911432 --> n8a6e2af1
  n4aafa92c --> n421e0ec3
  nc4eb7a45 --> n44911432
  n8c234a81 --> na5bb52f8
  ne4e0c430 --> n6e1621cd
  n44911432 --> nf9026078
  na1e3894b --> nc5297c1b
  n2cc6ed75 --> n1aafa816
  n44911432 --> nd457c3be
  n44911432 --> n4fe8f20e
  n210add96 --> n4d3ec133
  n44911432 --> nad734095
  n35e752f4 --> ne4e0c430
  n2c9d29bd --> ne4e0c430
  n4aafa92c --> n4d3ec133
  nde48e1fc --> n0e209692
  n8986b069 --> nfe7a6c1f
  nd457c3be --> n6cabbeab
  n44911432 --> n4aafa92c
  n6e1621cd --> n3aaed87c
  ne22b5687 --> n2a61d9e1
  n4d3ec133 --> nfe7a6c1f
  n4d3ec133 --> nfe9a5757
  n44911432 --> n48382513
  n5360779a --> n6a01f22c
  n4a843183 --> n6f1f3592
  n4fe8f20e --> n6fddc4f7
  n44911432 --> n1bfb2fb3
  ne0470225 --> n2c9d29bd
  n44911432 --> n10098bbf
  nd457c3be --> n421e0ec3
  ne4e0c430 --> na5bb52f8
  n8c234a81 --> ne4e0c430
  ne22b5687 --> n210add96
  ne4e0c430 --> nfe7a6c1f
  n48382513 --> n4d3ec133
  n8a6e2af1 --> n8372a34c
  n031610c4 --> n7ba033e3
  n3aaed87c --> n45dd0f5b
  n6f1f3592 --> n5eacdfde
  n48382513 --> n210add96
  n6fddc4f7 --> n4dc5810c
  n4dc5810c --> n4a843183
  n3e0b74e7 --> naea140e4
  n44911432 --> n858cac69
  ne4e0c430 --> nde48e1fc
  ncac932c5 --> na1e3894b
  n6f1f3592 --> nde48e1fc
  n5360779a --> nde6eadd5
  n1aafa816 --> nb16a055c
  n5360779a --> nfe9a5757
  n35e752f4 --> n88d4c03d
  nb16a055c --> n7ba033e3
  n44911432 --> n5360779a
  n1bfb2fb3 --> na95f5629
  ne4e0c430 --> n5215a49f
  n1bfb2fb3 --> ne0470225
  n8c234a81 --> n88d4c03d
  ne4e0c430 --> ncac932c5
  n4d3ec133 --> nbb4bb303
  n44911432 --> nb0f6ada0
  naea140e4 --> n35e752f4
  n8986b069 --> n0e209692
  n2c9d29bd --> n5eacdfde
  nbb4bb303 --> n56a67c43
  n6f1f3592 --> n3e0b74e7
  ne4e0c430 --> nde6eadd5
  na95f5629 --> ne0470225
  nd457c3be --> ne22b5687
  ne0470225 --> n8c234a81
  ne4e0c430 --> nfe9a5757
```

## Legend

- Rectangle = code/compute block (`type: 1`)
- Stadium = markdown/note block (`type: 4`)
- Hexagon = LLM/agent block (`type: 9`)
