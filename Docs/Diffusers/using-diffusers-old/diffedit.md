# DiffEdit

تتيح لك أداة DiffEdit تحرير الصور دون الحاجة إلى إنشاء قناع يدويًا. حيث تقوم تلقائيًا بتوليد القناع بناءً على استعلام نصي، مما يجعل إنشاء القناع أسهل بشكل عام دون الحاجة إلى برامج تحرير الصور. تعمل خوارزمية DiffEdit في ثلاث خطوات:

1. يقوم نموذج الانتشار بإزالة الضوضاء من صورة ما بناءً على نص استعلام ونص مرجعي، مما يؤدي إلى تقديرات ضوضاء مختلفة لمناطق مختلفة من الصورة؛ ويتم استخدام الفرق لاستنتاج قناع لتحديد أي جزء من الصورة يحتاج إلى تغيير ليتطابق مع نص الاستعلام.

2. يتم ترميز الصورة المدخلة إلى مساحة الكامنة باستخدام DDIM.

3. يتم فك ترميز الكامنات باستخدام نموذج الانتشار المشروط على نص الاستعلام، باستخدام القناع كدليل بحيث تظل البكسلات خارج القناع كما هي في الصورة المدخلة.

سيوضح هذا الدليل كيفية استخدام DiffEdit لتحرير الصور دون إنشاء قناع يدويًا.

قبل البدء، تأكد من تثبيت المكتبات التالية:

```py
# قم بإلغاء التعليق لتثبيت المكتبات الضرورية في Colab
#! pip install -q diffusers transformers accelerate
```

يتطلب [`StableDiffusionDiffEditPipeline`] قناع صورة ومجموعة من الكامنات المعكوسة جزئيًا. يتم إنشاء قناع الصورة من الدالة [`~StableDiffusionDiffEditPipeline.generate_mask`]، ويتضمن معلمتين، `source_prompt` و`target_prompt`. تحدد هذه المعلمات ما سيتم تحريره في الصورة. على سبيل المثال، إذا كنت تريد تغيير وعاء من *الفواكه* إلى وعاء من *الكمثرى*، فستكون:

```py
source_prompt = "وعاء من الفواكه"
target_prompt = "وعاء من الكمثرى"
```

تتم توليد الكامنات المعكوسة جزئيًا من الدالة [`~StableDiffusionDiffEditPipeline.invert`]، ومن الجيد عمومًا تضمين `prompt` أو *caption* لوصف الصورة للمساعدة في توجيه عملية أخذ العينات العكسية للكامن. غالبًا ما يكون التعليق هو `source_prompt` الخاص بك، ولكن يمكنك تجربة أوصاف نصية أخرى!

قم بتحميل الأنبوب، والمجدول، والمجدول العكسي، وتمكين بعض التحسينات لتقليل استخدام الذاكرة:

```py
استيراد الشعلة
من الناشرين استيراد DDIMScheduler، DDIMInverseScheduler، StableDiffusionDiffEditPipeline

الأنبوب = StableDiffusionDiffEditPipeline.from_pretrained (
"stabilityai/stable-diffusion-2-1"،
torch_dtype=torch.float16،
safety_checker=None،
use_safetensors=True،
)
pipeline.scheduler = DDIMScheduler.from_config (pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config (pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()
```

قم بتحميل الصورة التي تريد تحريرها:

```py
من utils utils استيراد load_image، make_image_grid

img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
الصورة الخام = تحميل الصورة (img_url). resize ((768، 768))
الصورة الخام
```

استخدم دالة [`~StableDiffusionDiffEditPipeline.generate_mask`] لتوليد قناع الصورة. ستحتاج إلى تمرير `source_prompt` و`target_prompt` لتحديد ما سيتم تحريره في الصورة:

```py
من PIL استيراد صورة

source_prompt = "وعاء من الفواكه"
target_prompt = "سلة من الكمثرى"
mask_image = pipeline.generate_mask (
الصورة = الصورة الخام،
source_prompt=source_prompt،
target_prompt=target_prompt،
)
صورة.fromarray ((mask_image.squeeze()* 255).astype ("uint8")، "L"). resize ((768، 768))
```

بعد ذلك، قم بإنشاء الكامنات المعكوسة ومرر لها تعليقًا يصف الصورة:

```py
inv_latents = pipeline.invert (prompt=source_prompt، image=raw_image). latents
```

أخيرًا، قم بتمرير قناع الصورة والكامنات المعكوسة إلى الأنبوب. يصبح `target_prompt` الآن `prompt`، ويتم استخدام `source_prompt` كـ `negative_prompt`:

```py
output_image = pipeline (
prompt=target_prompt،
mask_image=mask_image،
image_latents=inv_latents،
negative_prompt=source_prompt،
).images [0]
mask_image = Image.fromarray ((mask_image.squeeze()* 255).astype ("uint8")، "L"). resize ((768، 768))
make_image_grid ([الصورة الخام، mask_image، output_image]، الصفوف=1، cols=3)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://github.com/Xiang-cd/DiffEdit-stable-diffusion/blob/main/assets/target.png?raw=true"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المعدلة</figcaption>
</div>
</div>

## إنشاء تضمين المصدر والهدف

يمكن إنشاء تضمينات المصدر والهدف تلقائيًا باستخدام نموذج [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) بدلاً من إنشائها يدويًا.

قم بتحميل نموذج Flan-T5 والمحلل اللغوي من مكتبة 🤗 Transformers:

```py
استيراد الشعلة
من المحولات استيراد AutoTokenizer، T5ForConditionalGeneration

المحلل اللغوي = AutoTokenizer.from_pretrained ("google/flan-t5-large")
النموذج = T5ForConditionalGeneration.from_pretrained ("google/flan-t5-large"، device_map="auto"، torch_dtype=torch.float16)
```

قدم بعض النصوص الأولية لطلب النموذج لتوليد مطالبات المصدر والهدف.

```py
source_concept = "bowl"
target_concept = "basket"

source_text = f "قدم تعليقًا للصور التي تحتوي على {source_concept}. "
"يجب أن تكون التعليقات باللغة الإنجليزية وألا يتجاوز طولها 150 حرفًا."

target_text = f "قدم تعليقًا للصور التي تحتوي على {target_concept}. "
"يجب أن تكون التعليقات باللغة الإنجليزية وألا يتجاوز طولها 150 حرفًا."
```

بعد ذلك، قم بإنشاء دالة مساعدة لتوليد المطالبات:

```py
@ torch.no_grad ()
def generate_prompts (input_prompt):
input_ids = tokenizer (input_prompt، return_tensors="pt"). input_ids.to ("cuda")

outputs = model.generate (
input_ids، temperature=0.8، num_return_sequences=16، do_sample=True، max_new_tokens=128، top_k=10
)
return tokenizer.batch_decode (outputs، skip_special_tokens=True)

source_prompts = generate_prompts (source_text)
target_prompts = generate_prompts (target_text)
طباعة source_prompts
طباعة target_prompts
```

<Tip>
تحقق من دليل [استراتيجية التوليد](https://huggingface.co/docs/transformers/main/en/generation_strategies) إذا كنت مهتمًا بمعرفة المزيد عن استراتيجيات توليد نص مختلف الجودة.
</Tip>

قم بتحميل نموذج الترميز النصي المستخدم بواسطة [`StableDiffusionDiffEditPipeline`] لترميز النص. ستستخدم برنامج الترميز النصي لحساب التضمينات النصية:

```py
استيراد الشعلة
من الناشرين استيراد StableDiffusionDiffEditPipeline

الأنبوب = StableDiffusionDiffEditPipeline.from_pretrained (
"stabilityai/stable-diffusion-2-1"، torch_dtype=torch.float16، use_safetensors=True
)
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

@ torch.no_grad ()
def embed_prompts (الجمل، المحلل اللغوي، text_encoder، device="cuda"):
التضمينات = []
for sent in sentences:
text_inputs = tokenizer (
sent،
padding="max_length"،
max_length=tokenizer.model_max_length،
truncation=True،
return_tensors="pt"،
)
text_input_ids = text_inputs.input_ids
prompt_embeds = text_encoder (text_input_ids.to (device)، attention_mask=None) [0]
التضمينات.append (prompt_embeds)
return torch.concatenate (التضمينات، dim=0). mean (dim=0). unsqueeze (0)

source_embeds = embed_prompts (source_prompts، pipeline.tokenizer، pipeline.text_encoder)
target_embeds = embed_prompts (target_prompts، pipeline.tokenizer، pipeline.text_encoder)
```

أخيرًا، قم بتمرير التضمينات إلى دالات [`~StableDiffusionDiffEditPipeline.generate_mask`] و [`~StableDiffusionDiffEditPipeline.invert`]، والأنبوب لتوليد الصورة:

```diff
من الناشرين استيراد DDIMInverseScheduler، DDIMScheduler
من utils utils استيراد load_image، make_image_grid
من PIL استيراد صورة

pipeline.scheduler = DDIMScheduler.from_config (pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config (pipeline.scheduler.config)

img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
الصورة الخام = تحميل الصورة (img_url). resize ((768، 768))

mask_image = pipeline.generate_mask (
الصورة = الصورة الخام،
-     source_prompt=source_prompt،
-     target_prompt=target_prompt،
+     source_prompt_embeds=source_embeds،
+     target_prompt_embeds=target_embeds،
)

inv_latents = pipeline.invert (
-     prompt=source_prompt،
+     prompt_embeds=source_embeds،
الصورة = الصورة الخام،
). latents

output_image = pipeline (
mask_image=mask_image،
image_latents=inv_latents،
-     prompt=target_prompt،
-     negative_prompt=source_prompt،
+     prompt_embeds=target_embeds،
+     negative_prompt_embeds=source_embeds،
).images [0]
mask_image = Image.fromarray ((mask_image.squeeze()* 255).astype ("uint8")، "L")
make_image_grid ([الصورة الخام، mask_image، output_image]، الصفوف=1، cols=3)
```
بالتأكيد! هذا هو النص المترجم وفقًا لتعليماتك:

## إنشاء عنوان توضيحي للانعكاس
يمكنك استخدام `source_prompt` كعنوان توضيحي للمساعدة في إنشاء الصور المخفية جزئيًا، أو يمكنك أيضًا استخدام نموذج [BLIP](https://huggingface.co/docs/transformers/model_doc/blip) لإنشاء عنوان توضيحي تلقائيًا.

قم بتحميل نموذج BLIP ومعالجته من مكتبة 🤗 Transformers:
```py
# لا تترجم هذا الكود البرمجي
```
قم بإنشاء دالة فائدة لإنشاء عنوان توضيحي من صورة الإدخال:
```py
# لا تترجم هذا الكود البرمجي
```
قم بتحميل صورة إدخال وإنشاء عنوان توضيحي لها باستخدام دالة `generate_caption`:
```py
# لا تترجم هذا الكود البرمجي
```
<div class="flex justify-center">
<figure>
<img class="rounded-xl" src="https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"/>
<figcaption class="text-center">العنوان المولد: "صورة فوتوغرافية لفاكهة في وعاء على طاولة"</figcaption>
</figure>
</div>

الآن، يمكنك وضع العنوان التوضيحي في دالة [`~StableDiffusionDiffEditPipeline.invert`] لإنشاء الصور المخفية جزئيًا!