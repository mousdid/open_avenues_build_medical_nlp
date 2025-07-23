from metaflow import FlowSpec, step, card, current
from metaflow.cards import Markdown, Image






class ReportsFlow(FlowSpec):

    @step
    def start(self):
        print("Starting the medical reports flow")
        self.next(self.eda)

    @card
    @step
    def eda(self):
        from utils.EDA import (
    load_and_preview_dataset,
    text_length_insights,
    most_frequent_stopwords,
    most_frequent_words,
    get_top_ngrams,
    show_wordcloud,
    preprocessing,
    fig_to_bytes
)
        print("Performing exploratory data analysis...")
        self.reports, self.shape, self.head = load_and_preview_dataset("../data/downloads/open_ave_data.csv")
        current.card.append(Markdown(f"### ✅ Loaded dataset with shape: `{self.shape}`"))
        current.card.append(Markdown("#### Preview of first rows:"))
        current.card.append(Markdown(f"```\n{self.head.to_markdown()}\n```"))

        # Preprocessing: drop missing rows
        cleaned_reports, dropped_count = preprocessing(self.reports)
        current.card.append(Markdown(f"#### 🧹 Rows dropped due to missing values: `{dropped_count}`"))

        # Text length insights
        figs = text_length_insights(cleaned_reports)
        current.card.append(Markdown("### 📊 Text Length Insights"))
        for fig in figs:
            current.card.append(Image(src=fig_to_bytes(fig)))

        # Most frequent stopwords
        top_stop, fig_stop = most_frequent_stopwords(cleaned_reports)
        current.card.append(Markdown("### 🛑 Top 10 Stopwords"))
        current.card.append(Markdown(f"```\n{top_stop}\n```"))
        current.card.append(Image(src=fig_to_bytes(fig_stop)))

        # Most frequent non-stopwords
        words_info, (fig_stop2, fig_nonstop) = most_frequent_words(cleaned_reports)
        current.card.append(Markdown("### 🔤 Top 40 Non-Stopwords"))
        current.card.append(Markdown(f"```\n{words_info['non_stopwords']}\n```"))
        current.card.append(Image(src=fig_to_bytes(fig_nonstop)))

        # Top bigrams and trigrams
        bigrams, fig_bigrams = get_top_ngrams(cleaned_reports, n=2, top_k=10)
        trigrams, fig_trigrams = get_top_ngrams(cleaned_reports, n=3, top_k=10)
        current.card.append(Markdown("### 📛 Top 10 Bigrams"))
        current.card.append(Markdown(f"```\n{bigrams}\n```"))
        current.card.append(Image(src=fig_to_bytes(fig_bigrams)))
        current.card.append(Markdown("### 🔺 Top 10 Trigrams"))
        current.card.append(Markdown(f"```\n{trigrams}\n```"))
        current.card.append(Image(src=fig_to_bytes(fig_trigrams)))

        # Word cloud
        fig_wc = show_wordcloud(cleaned_reports)
        current.card.append(Markdown("### ☁️ Word Cloud"))
        current.card.append(Image(src=fig_to_bytes(fig_wc)))

        self.reports = cleaned_reports
        self.next(self.prompt_extraction, self.finetune_extraction)

    @step
    def prompt_extraction(self):
        from utils.prompting import (
    extract_metadata_with_labels,
    build_few_shots_prompt,
    build_chat_messages,
    run_extraction_qwen,
    batch_extraction
)
        print("Extracting relevant fields using prompted LLM...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import pandas as pd

        # Prepare few-shot examples (first 30 rows)
        few_shot_df = self.reports[['ReportText', 'ExamName', 'clinicaldata', 'findings', 'impression']].head(30)
        # Load model and tokenizer (CPU only)
        model_id = "Qwen/Qwen1.5-0.5B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)  # No .to("cuda")

        # Run batch extraction for a range of indices 
        start_index = 31
        end_index = 50
        predicted_df = batch_extraction(self.reports, few_shot_df, tokenizer, model, start_index, end_index)

        # Ground truth
        true_df = self.reports.iloc[start_index:end_index][['ExamName', 'clinicaldata', 'findings', 'impression']].reset_index(drop=True)
        combined_df = pd.concat([true_df, predicted_df.add_suffix("_pred")], axis=1)

        self.prompt_output = combined_df  # Save the DataFrame for later analysis
        print("Prompt extraction completed.")
        self.next(self.join)

    @card
    @step
    def finetune_extraction(self):
        from utils.finetuning import run_finetuning
        print("Extracting relevant fields using fine-tuned LLM...")
        # Use the cleaned reports from EDA
        result = run_finetuning(self.reports)
        self.finetune_output = result['report']
        current.card.append(Markdown("### 🏋️‍♂️ Fine-tuning Classification Report"))
        current.card.append(Markdown(f"```\n{self.finetune_output}\n```"))
        self.next(self.join)

    @step
    def join(self, inputs):
        print("Joining outputs from both paths...")
        self.prompt_output = inputs.prompt_extraction.prompt_output
        self.finetune_output = inputs.finetune_extraction.finetune_output
        self.next(self.result_analysis)

    @step
    def result_analysis(self):
        print("Analyzing results...")
        current.card.append(Markdown("## 🔎 Results Analysis"))
        current.card.append(Markdown("### Prompt Extraction Results"))
        current.card.append(Markdown(f"```\n{self.prompt_output}\n```"))
        current.card.append(Markdown("### Finetune Extraction Results"))
        current.card.append(Markdown(f"```\n{self.finetune_output}\n```"))
        self.next(self.end)

    @step
    def end(self):
        print("✅ Medical report extraction completed.")

if __name__ == "__main__":
    ReportsFlow()
