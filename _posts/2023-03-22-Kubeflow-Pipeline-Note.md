## High Level Overview of Using Kubeflow Pipeline

To use KFP

1. Author a pipeline with one or more components using python sdk's domail specific library (DSL)
2. Compile the pipeline into static YAML file by using DSL's compiler
3. Submit your pipelien to run in the KFP backend
4. View your runs, experiments, and ML artifacts on the KFP Dashboard.

### Quick Demo

**Step 1**
```Python
export PIPELINE_VERSION="2.0.0-alpha.4"

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```

**Step2**





