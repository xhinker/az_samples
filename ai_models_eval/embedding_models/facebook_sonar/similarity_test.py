#%%
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model

blaser_ref = load_blaser_model("blaser_2_0_ref").eval()
blaser_qe = load_blaser_model("blaser_2_0_qe").eval()
text_embedder = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder")

#%%
src_embs = text_embedder.predict(["Le chat s'assit sur le tapis."], source_lang="fra_Latn")
ref_embs = text_embedder.predict(["The cat sat on the mat."], source_lang="eng_Latn")
mt_embs = text_embedder.predict(["The cat sat down on the carpet."], source_lang="eng_Latn")

print(blaser_ref(src=src_embs, ref=ref_embs, mt=mt_embs).item())  # 4.688
print(blaser_qe(src=src_embs, mt=mt_embs).item())  # 4.708