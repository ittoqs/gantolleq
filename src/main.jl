module Gantolleq

using ArgParse

include("utils.jl")
include("trained_engine.jl")
include("pretrained_engine.jl")

function main(args_in = ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--mode"
            help = "trained | predict"
            required = true

        "--engine"
            help = "trained | pretrained (untuk mode predict)"
            default = "trained"

        "--data"
            help = "CSV dataset latih (untuk mode trained)"
            required = false

        "--model"
            help = "Path model untuk engine trained"
            default = "models/trained_model.bson"

        "--input"
            help = "Path input teks: .csv (kolom text) atau .jsonl ({text:...})"
            required = false

        "--out_prefix"
            help = "Prefix output (akan menghasilkan .csv dan .json)"
            required = true

        "--pretrained_model"
            help = "HuggingFace model id untuk engine pretrained (download lokal)"
            default = "cardiffnlp/twitter-roberta-base-sentiment-latest"

        "--thresholds"
            help = "pos_strong=0.7,neg_strong=0.7"
            default = "pos_strong=0.7,neg_strong=0.7"
    end

    args = parse_args(args_in, s)
    mode = args["mode"]

    if mode == "trained"
        if isnothing(args["data"])
            error("--data harus diisi untuk mode trained")
        end
        data_path = args["data"]
        model_path = args["model"]
        train_from_csv(data_path, model_path)
        println("Training selesai. Model: $model_path")
        return
    end

    if mode == "predict"
        if isnothing(args["input"])
            error("--input harus diisi untuk mode predict")
        end
        engine = args["engine"]
        input_path = args["input"]
        out_prefix = args["out_prefix"]
        thresholds = parse_thresholds(args["thresholds"])

        if engine == "trained"
            predict_trained(input_path, args["model"], out_prefix)
        elseif engine == "pretrained"
            predict_pretrained(input_path, args["pretrained_model"], thresholds, out_prefix)
        else
            error("engine tidak dikenal: $engine")
        end
        return
    end

    error("mode tidak dikenal: $mode")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
