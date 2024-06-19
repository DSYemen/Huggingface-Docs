# 🤗 Transformers

أحدث ما توصلت إليه الآلة في مجال التعلم لبيئة [PyTorch](https://pytorch.org/) و [TensorFlow](https://www.tensorflow.org/) و [JAX](https://jax.readthedocs.io/en/latest/) .

🤗 يوفر Transformers واجهات برمجة تطبيقات (APIs) وأدوات للتنزيل والتدريب السهل لنماذج مسبقة التدريب. يمكن أن يقلل استخدام النماذج مسبقة التدريب من تكاليف الحوسبة والبصمة الكربونية لديك، ويوفر لك الوقت والموارد اللازمة لتدريب نموذج من الصفر. تدعم هذه النماذج المهام الشائعة في أوضاع مختلفة، مثل:

📝 **معالجة اللغات الطبيعية**: تصنيف النصوص، وتعريف الكيانات المسماة، والرد على الأسئلة، ونمذجة اللغة، والتلخيص، والترجمة، والاختيار من متعدد، وتوليد النصوص. <br>
🖼️ **الرؤية الحاسوبية**: تصنيف الصور، وكشف الأجسام، والتجزئة. <br>
🗣️ **الصوت**: التعرف التلقائي على الكلام، وتصنيف الصوت. <br>
🐙 **متعدد الوسائط**: الرد على الأسئلة الجدولية، والتعرف البصري على الحروف، واستخراج المعلومات من المستندات الممسوحة ضوئيًا، وتصنيف الفيديو، والرد على الأسئلة البصرية.

يدعم 🤗 Transformers قابلية التشغيل البيني للإطار بين PyTorch و TensorFlow و JAX. يوفر هذا المرونة لاستخدام إطار عمل مختلف في كل مرحلة من مراحل حياة النموذج؛ قم بتدريب نموذج في ثلاث خطوط من التعليمات البرمجية في إطار واحد، وقم بتحميله للاستدلال في إطار آخر. يمكن أيضًا تصدير النماذج إلى تنسيق مثل ONNX و TorchScript للنشر في بيئات الإنتاج.

انضم إلى المجتمع المتنامي على [Hub](https://huggingface.co/models) أو [المنتدى](https://discuss.huggingface.co/) أو [Discord](https://discord.com/invite/JfAtkvEtRb) اليوم!

## إذا كنت تبحث عن دعم مخصص من فريق Hugging Face

<a target="_blank" href="https://huggingface.co/support">
<img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a>

## المحتويات

تنقسم الوثائق إلى خمسة أقسام:

- **ابدأ** يقدم جولة سريعة في المكتبة وتعليمات التثبيت للبدء.

- **الدروس التعليمية** هي مكان رائع للبدء إذا كنت مبتدئًا. سيساعدك هذا القسم على اكتساب المهارات الأساسية التي تحتاجها للبدء في استخدام المكتبة.

- **أدلة كيفية الاستخدام** توضح لك كيفية تحقيق هدف محدد، مثل ضبط نموذج مسبق التدريب لنمذجة اللغة أو كيفية كتابة ومشاركة نموذج مخصص.

- **الأدلة المفاهيمية** تقدم مناقشة وشرحًا أكثر للمفاهيم والأفكار الأساسية التي تقوم عليها النماذج والمهام وفلسفة التصميم في 🤗 Transformers.

- **واجهة برمجة التطبيقات** تصف جميع الفئات والوظائف:

   - **الفئات الرئيسية** تشرح الفئات الأكثر أهمية مثل التكوين والنماذج ومعالج المدخلات والأنابيب.

   - **النماذج** تشرح الفئات والوظائف المتعلقة بكل نموذج يتم تنفيذه في المكتبة.

   - **مساعدون داخليون** يوضحون فئات ووظائف المساعدة المستخدمة داخليًا.
## النماذج والأطر المدعومة
يمثل الجدول أدناه الدعم الحالي في المكتبة لكل من هذه النماذج، سواء كان لديها محلل نحوي Python (يسمى "البطيء"). محلل نحوي "سريع" مدعوم من مكتبة 🤗 Tokenizers، وما إذا كان لديها دعم في Jax (عبر Flax) و/أو PyTorch و/أو TensorFlow.

| النموذج | دعم PyTorch | دعم TensorFlow | دعم Flax |
|:--------:|:------------:|:--------------:|:---------:|
|  [ALBERT](model_doc/albert)  |      ✅       |        ✅       |     ✅     |
|   [ALIGN](model_doc/align)   |      ✅       |        ❌       |     ❌     |
| [AltCLIP](model_doc/altclip) |      ✅       |        ❌       |     ❌     |
| [Audio Spectrogram Transformer](model_doc/audio-spectrogram-transformer) |      ✅       |        ❌       |     ❌     |
|  [Autoformer](model_doc/autoformer)  |      ✅       |        ❌       |     ❌     |
|    [Bark](model_doc/bark)    |      ✅       |        ❌       |     ❌     |
|    [BART](model_doc/bart)    |      ✅       |        ✅       |     ✅     |
|   [BARThez](model_doc/barthez)   |      ✅       |        ✅       |     ✅     |
|   [BARTpho](model_doc/bartpho)   |      ✅       |        ✅       |     ✅     |
|    [BEiT](model_doc/beit)    |      ✅       |        ❌       |     ✅     |
|    [BERT](model_doc/bert)    |      ✅       |        ✅       |     ✅     |
| [Bert Generation](model_doc/bert-generation) |      ✅       |        ❌       |     ❌     |
|   [BertJapanese](model_doc/bert-japanese)   |      ✅       |        ✅       |     ✅     |
|    [BERTweet](model_doc/bertweet)    |      ✅       |        ✅       |     ✅     |
|    [BigBird](model_doc/big_bird)    |      ✅       |        ❌       |     ✅     |
| [BigBird-Pegasus](model_doc/bigbird_pegasus) |      ✅       |        ❌       |     ❌     |
|     [BioGpt](model_doc/biogpt)     |      ✅       |        ❌       |     ❌     |
|       [BiT](model_doc/bit)       |      ✅       |        ❌       |     ❌     |
|    [Blenderbot](model_doc/blenderbot)    |      ✅       |        ✅       |     ✅     |
|  [BlenderbotSmall](model_doc/blenderbot-small)   |      ✅       |        ✅       |     ✅     |
|      [BLIP](model_doc/blip)      |      ✅       |        ✅       |     ❌     |
|        [BLIP-2](model_doc/blip-2)        |      ✅       |        ❌       |     ❌     |
|       [BLOOM](model_doc/bloom)       |      ✅       |        ❌       |     ✅     |
|      [BORT](model_doc/bort)      |      ✅       |        ✅       |     ✅     |
|   [BridgeTower](model_doc/bridgetower)   |      ✅       |        ❌       |     ❌     |
|      [BROS](model_doc/bros)      |      ✅       |        ❌       |     ❌     |
|      [ByT5](model_doc/byt5)      |      ✅       |        ✅       |     ✅     |
|   [CamemBERT](model_doc/camembert)   |      ✅       |        ✅       |     ❌     |
|      [CANINE](model_doc/canine)      |      ✅       |        ❌       |     ❌     |
|    [Chinese-CLIP](model_doc/chinese_clip)    |      ✅       |        ❌       |     ❌     |
|      [CLAP](model_doc/clap)      |      ✅       |        ❌       |     ❌     |
|      [CLIP](model_doc/clip)      |      ✅       |        ✅       |     ✅     |
|     [CLIPSeg](model_doc/clipseg)     |      ✅       |        ❌       |     ❌     |
|      [CLVP](model_doc/clvp)      |      ✅       |        ❌       |     ❌     |
|      [CodeGen](model_doc/codegen)      |      ✅       |        ❌       |     ❌     |
|   [CodeLlama](model_doc/code_llama)   |      ✅       |        ❌       |     ✅     |
|      [Cohere](model_doc/cohere)      |      ✅       |        ❌       |     ❌     |
| [Conditional DETR](model_doc/conditional_detr) |      ✅       |        ❌       |     ❌     |
|    [ConvBERT](model_doc/convbert)    |      ✅       |        ✅       |     ❌     |
|    [ConvNeXT](model_doc/convnext)    |      ✅       |        ✅       |     ❌     |
|      [ConvNeXTV2](model_doc/convnextv2)      |      ✅       |        ✅       |     ❌     |
|       [CPM](model_doc/cpm)       |      ✅       |        ✅       |     ✅     |
|     [CPM-Ant](model_doc/cpmant)     |      ✅       |        ❌       |     ❌     |
|      [CTRL](model_doc/ctrl)      |      ✅       |        ✅       |     ❌     |
|       [CvT](model_doc/cvt)       |      ✅       |        ✅       |     ❌     |
|   [Data2VecAudio](model_doc/data2vec)   |      ✅       |        ❌       |     ❌     |
|    [Data2VecText](model_doc/data2vec)    |      ✅       |        ❌       |     ❌     |
|   [Data2VecVision](model_doc/data2vec)   |      ✅       |        ✅       |     ❌     |
|      [DBRX](model_doc/dbrx)      |      ✅       |        ❌       |     ❌     |
|     [DeBERTa](model_doc/deberta)     |      ✅       |        ✅       |     ❌     |
|      [DeBERTa-v2](model_doc/deberta-v2)      |      ✅       |        ✅       |     ❌     |
| [Decision Transformer](model_doc/decision_transformer) |      ✅       |        ❌       |     ❌     |
|  [Deformable DETR](model_doc/deformable_detr)   |      ✅       |        ❌       |     ❌     |
|      [DeiT](model_doc/deit)      |      ✅       |        ✅       |     ❌     |
|      [DePlot](model_doc/deplot)      |      ✅       |        ❌       |     ❌     |
|  [Depth Anything](model_doc/depth_anything)   |      ✅       |        ❌       |     ❌     |
|      [DETA](model_doc/deta)      |      ✅       |        ❌       |     ❌     |
|      [DETR](model_doc/detr)      |      ✅       |        ❌       |     ❌     |
|    [DialoGPT](model_doc/dialogpt)    |      ✅       |        ✅       |     ✅     |
|      [DiNAT](model_doc/dinat)      |      ✅       |        ❌       |     ❌     |
|      [DINOv2](model_doc/dinov2)      |      ✅       |        ❌       |     ❌     |
|   [DistilBERT](model_doc/distilbert)   |      ✅       |        ✅       |     ✅     |
|       [DiT](model_doc/dit)       |      ✅       |        ❌       |     ✅     |
|     [DonutSwin](model_doc/donut)     |      ✅       |        ❌       |     ❌     |
|       [DPR](model_doc/dpr)       |      ✅       |        ✅       |     ❌     |
|       [DPT](model_doc/dpt)       |      ✅       |        ❌       |     ❌     |
|   [EfficientFormer](model_doc/efficientformer)   |      ✅       |        ✅       |     ❌     |
|  [EfficientNet](model_doc/efficientnet)   |      ✅       |        ❌       |     ❌     |
|    [ELECTRA](model_doc/electra)    |      ✅       |        ✅       |     ✅     |
|    [EnCodec](model_doc/encodec)    |      ✅       |        ❌       |     ❌     |
| [Encoder decoder](model_doc/encoder-decoder) |      ✅       |        ✅       |     ✅     |
|      [ERNIE](model_doc/ernie)      |      ✅       |        ❌       |     ❌     |
|      [ErnieM](model_doc/ernie_m)      |      ✅       |        ❌       |     ❌     |
|        [ESM](model_doc/esm)        |      ✅       |        ✅       |     ❌     |
|   [FairSeq Machine-Translation](model_doc/fsmt)   |      ✅       |        ❌       |     ❌     |
|      [Falcon](model_doc/falcon)      |      ✅       |        ❌       |     ❌     |
| [FastSpeech2Conformer](model_doc/fastspeech2_conformer) |      ✅       |        ❌       |     ❌     |
|     [FLAN-T5](model_doc/flan-t5)     |      ✅       |        ✅       |     ✅     |
|    [FLAN-UL2](model_doc/flan-ul2)    |      ✅       |        ✅       |     ✅     |
|    [FlauBERT](model_doc/flaubert)    |      ✅       |        ✅       |     ❌     |
|      [FLAVA](model_doc/flava)      |      ✅       |        ❌       |     ❌     |
|       [FNet](model_doc/fnet)       |      ✅       |        ❌       |     ❌     |
|     [FocalNet](model_doc/focalnet)     |      ✅       |        ❌       |     ❌     |
| [Funnel Transformer](model_doc/funnel) |      ✅       |        ✅       |     ❌     |
|      [Fuyu](model_doc/fuyu)      |      ✅       |        ❌       |     ❌     |
|      [Gemma](model_doc/gemma)      |      ✅       |        ❌       |     ✅     |
|        [GIT](model_doc/git)        |      ✅       |        ❌       |     ❌     |
|       [GLPN](model_doc/glpn)       |      ✅       |        ❌       |     ❌     |
|     [GPT Neo](model_doc/gpt_neo)     |      ✅       |        ❌       |     ✅     |
|    [GPT NeoX](model_doc/gpt_neox)    |      ✅       |        ❌       |     ❌     |
| [GPT NeoX Japanese](model_doc/gpt_neox_japanese) |      ✅       |        ❌       |     ❌     |
|      [GPT-J](model_doc/gptj)      |      ✅       |        ✅       |     ✅     |
|     [GPT-Sw3](model_doc/gpt-sw3)     |      ✅       |        ✅       |     ✅     |
|   [GPTBigCode](model_doc/gpt_bigcode)   |      ✅       |        ❌       |     ❌     |
|   [GPTSAN-japanese](model_doc/gptsan-japanese)   |      ✅       |        ❌       |     ❌     |
|    [Graphormer](model_doc/graphormer)    |      ✅       |        ❌       |     ❌     |
|  [Grounding DINO](model_doc/grounding-dino)   |      ✅       |        ❌       |     ❌     |
|     [GroupViT](model_doc/groupvit)     |      ✅       |        ✅       |     ❌     |
|     [HerBERT](model_doc/herbert)     |      ✅       |        ✅       |     ✅     |
|      [Hubert](model_doc/hubert)      |      ✅       |        ✅       |     ❌     |
|      [I-BERT](model_doc/ibert)      |      ✅       |        ❌       |     ❌     |
|     [IDEFICS](model_doc/idefics)     |      ✅       |        ✅       |     ❌     |
|    [Idefics2](model_doc/idefics2)    |      ✅       |        ❌       |     ❌     |
|    [ImageGPT](model_doc/imagegpt)    |      ✅       |        ❌       |     ❌     |
|    [Informer](model_doc/informer)    |      ✅       |        ❌       |     ❌     |
|  [InstructBLIP](model_doc/instructblip)   |      ✅       |        ❌       |     ❌     |
|      [Jamba](model_doc/jamba)      |      ✅       |        ❌       |     ❌     |
|     [JetMoe](model_doc/jetmoe)     |      ✅       |        ❌       |     ❌     |
|      [Jukebox](model_doc/jukebox)      |      ✅       |        ❌       |     ❌     |
|     [KOSMOS-2](model_doc/kosmos-2)     |      ✅       |        ❌       |     ❌     |
|     [LayoutLM](model_doc/layoutlm)     |      ✅       |        ✅       |     ❌     |
|   [LayoutLMv2](model_doc/layoutlmv2)   |      ✅       |        ❌       |     ❌     |
|   [LayoutLMv3](model_doc/layoutlmv3)   |      ✅       |        ✅       |     ❌     |
|    [LayoutXLM](model_doc/layoutxlm)    |      ✅       |        ❌       |     ❌     |
|        [LED](model_doc/led)        |      ✅       |        ✅       |     ❌     |
|      [LeViT](model_doc/levit)      |      ✅       |        ❌       |     ❌     |
|       [LiLT](model_doc/lilt)       |      ✅       |        ❌       |     ❌     |
|      [LLaMA](model_doc/llama)      |      ✅       |        ❌       |     ✅     |
|      [Llama2](model_doc/llama2)      |      ✅       |        ❌       |     ✅     |
|      [Llama3](model_doc/llama3)      |      ✅       |        ❌       |     ✅     |
|       [LLaVa](model_doc/llava)       |      ✅       |        ❌       |     ❌     |
|    [LLaVA-NeXT](model_doc/llava_next)    |      ✅       |        ❌       |     ❌     |
|    [Longformer](model_doc/longformer)    |      ✅       |        ✅       |     ❌     |
|      [LongT5](model_doc/longt5)      |      ✅       |        ❌       |     ✅     |
|        [LUKE](model_doc/luke)        |      ✅       |        ❌       |     ❌     |
|      [LXMERT](model_doc/lxmert)      |      ✅       |        ✅       |     ❌     |
|      [M-CTC-T](model_doc/mctct)      |      ✅       |        ❌       |     ❌     |
|     [M2M100](model_doc/m2m_100)     |      ✅       |        ❌       |     ❌     |
|   [MADLAD-400](model_doc/madlad-400)   |      ✅       |        ✅       |     ✅     |
|      [Mamba](model_doc/mamba)      |      ✅       |        ❌       |     ❌     |
|      [Marian](model_doc/marian)      |      ✅       |        ✅       |     ✅     |
|    [MarkupLM](model_doc/markuplm)    |      ✅       |        ❌       |     ❌     |
|   [Mask2Former