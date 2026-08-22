using TextAnalysis
using MLJ

function train_from_csv(csv_path::AbstractString, model_path::AbstractString)
    df = CSV.read(csv_path, DataFrame)
    @assert hasproperty(df, :text)
    @assert hasproperty(df, :label)

    texts = String.(df.text)
    raw_labels = String.(df.label)
    labels = [normalize_label(l) for l in raw_labels]

    keep = .!=(labels, "")
    texts = texts[keep]
    labels = labels[keep]

    y = categorical(labels; levels=FINAL_LABELS)

    # TF-IDF
    vect = TextAnalysis.TfidfVectorizer()
    vect_fit = TextAnalysis.fit!(vect, texts)
    X = TextAnalysis.transform(vect_fit, texts)

    # Multi-class classifier: LogisticRegression (probabilitas)
    clf = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
    model = clf(lambda=0.01)

    mach = machine(model, X, y)
    fit!(mach)

    model_dir = dirname(model_path)
    if !isempty(model_dir)
        mkpath(model_dir)
    end

    BSON.@save model_path vect_fit mach
    return model_path
end

function predict_trained(input_path::AbstractString, model_path::AbstractString, out_prefix::AbstractString)
    texts = read_input_texts(input_path)

    d = BSON.load(model_path)
    vect_fit = d[:vect_fit]
    mach = d[:mach]

    X = TextAnalysis.transform(vect_fit, texts)

    # MLJ probabilitas
    probs = predict(mach, X)

    labels = String[]
    scores_per_class = Vector{Dict{String,Float64}}()

    for i in 1:length(texts)
        # probs[i] adalah distribusi; kita pakai pdf(dist, cls)
        dist = probs[i]
        sc = Dict{String,Float64}()
        for cls in FINAL_LABELS
            sc[cls] = pdf(dist, cls)
        end
        best_idx = argmax([sc[cls] for cls in FINAL_LABELS])
        push!(labels, FINAL_LABELS[best_idx])
        push!(scores_per_class, sc)
    end

    write_outputs(texts, labels, scores_per_class, out_prefix)
end
