# تقنيات التحفيز 

تعد المحفزات مهمة لأنها تصف ما تريد من نموذج الانتشار توليده. أفضل المحفزات هي تلك المفصلة والمحددة والمُبنية بشكل جيد لمساعدة النموذج على تحقيق رؤيتك. ولكن صياغة محفز رائع يستغرق وقتًا وجهدًا، وفي بعض الأحيان قد لا يكون ذلك كافيًا لأن اللغة والكلمات قد لا تكون دقيقة. هنا تحتاج إلى تعزيز محفزك بتقنيات أخرى، مثل تحسين المحفزات ووزن المحفزات، للحصول على النتائج المرجوة.

سيوضح هذا الدليل كيفية استخدام تقنيات المحفزات هذه لتوليد صور عالية الجودة بمجهود أقل وتعديل وزن كلمات رئيسية معينة في المحفز.

## هندسة المحفزات

> [!TIP]
> هذا ليس دليلًا شاملًا عن هندسة المحفزات، ولكنه سيساعدك على فهم الأجزاء الضرورية لمحفز جيد. نشجعك على الاستمرار في تجربة محفزات مختلفة ودمجها بطرق جديدة لمعرفة ما يناسبك. كلما كتبت محفزات أكثر، ستطور حدسًا لما ينجح وما لا ينجح!

تؤدي نماذج الانتشار الجديدة وظيفتها بشكل جيد إلى حد ما في توليد صور عالية الجودة من محفز أساسي، ولكن لا يزال من المهم إنشاء محفز مكتوب بشكل جيد للحصول على أفضل النتائج. فيما يلي بعض النصائح لكتابة محفز جيد:

1. ما هي وسيطة الصورة؟ هل هي صورة فوتوغرافية، أم لوحة زيتية، أم توضيح ثلاثي الأبعاد، أم شيء آخر؟
2. ما هو موضوع الصورة؟ هل هو شخص، أم حيوان، أم كائن، أم منظر؟
3. ما هي التفاصيل التي تريد رؤيتها في الصورة؟ هنا يمكنك أن تكون مبدعًا حقًا والاستمتاع بتجربة كلمات مختلفة لإضفاء الحياة على صورتك. على سبيل المثال، ما هو الإضاءة؟ ما هي الحالة المزاجية والجمالية؟ ما هو أسلوب الفن أو التوضيح الذي تبحث عنه؟ كلما كانت الكلمات التي تستخدمها أكثر تحديدًا ودقة، كان فهم النموذج لما تريد توليده أفضل.

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/plain-prompt.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">"صورة أريكة على شكل موز في غرفة المعيشة"</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/detail-prompt.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">"أريكة على شكل موز ذات لون أصفر زاهٍ تستقر في غرفة معيشة مريحة، ويحتوي انحناءها على مجموعة من الوسائد الملونة. وعلى الأرضية الخشبية، تضيف سجادة ذات نقوش لمسة من السحر الإيكلكتيكي، وتقف نبتة في وعاء في الزاوية، ممتدة نحو أشعة الشمس التي تتسلل من النوافذ"</figcaption>
</div>
</div>
تحسين الفواصل الزمنية باستخدام GPT2

تحسين الفواصل الزمنية هو تقنية لتحسين جودة الفواصل الزمنية بسرعة دون بذل الكثير من الجهد في بنائها. يستخدم أسلوبًا مثل GPT2 الذي تم تدريبه مسبقًا على فواصل Stable Diffusion النصية لتحسين الفواصل الزمنية تلقائيًا باستخدام كلمات رئيسية إضافية مهمة لإنشاء صور عالية الجودة.

تعمل التقنية عن طريق إنشاء قائمة بكلمات رئيسية محددة وإجبار النموذج على توليد تلك الكلمات لتعزيز الفاصل الزمني الأصلي. بهذه الطريقة، يمكن أن يكون فاصل الوقت الخاص بك "قطة" ويمكن لـ GPT2 تحسين الفاصل الزمني إلى "لقطة فيلم سينمائي لقطة لقطة تستمتع بأشعة الشمس على سطح في تركيا، مفصلة للغاية، وميزانية إنتاج عالية، وفيلم هوليوود، وشاشة سينمائية، ومزاجية، وملحمية، وجميلة، وحبيبات فيلم حادة التركيز جميلة مفصلة معقدة مذهلة ملحمية".

نصيحة: يجب عليك أيضًا استخدام LORA ضوضاء التعويض لتحسين التباين في الصور الساطعة والمظلمة وخلق إضاءة أفضل بشكل عام. متاح من [stabilityai/stable-diffusion-xl-base-1.0](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0).

ابدأ بتحديد بعض الأساليب وقائمة بالكلمات (يمكنك الاطلاع على قائمة أكثر شمولاً [بالكلمات](https://hf.co/LykosAI/GPT-Prompt-Expansion-Fooocus-v2/blob/main/positive.txt) و [الأساليب](https://github.com/lllyasviel/Fooocus/tree/main/sdxl_styles) التي يستخدمها Fooocus) لتحسين فاصل معين.

تعريف بعض الأساليب وقائمة الكلمات:

```py
styles = {
"cinematic": "cinematic film still of {prompt}, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain",
"anime": "anime artwork of {prompt}, anime style, key visual, vibrant, studio anime, highly detailed",
"photographic": "cinematic photo of {prompt}, 35mm photograph, film, professional, 4k, highly detailed",
"comic": "comic of {prompt}, graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
"lineart": "line art drawing {prompt}, professional, sleek, modern, minimalist, graphic, line art, vector graphics",
"pixelart": " pixel-art {prompt}, low-res, blocky, pixel art style, 8-bit graphics",
}

words = [
"aesthetic", "astonishing", "beautiful", "breathtaking", "composition", "contrasted", "epic", "moody", "enhanced",
"exceptional", "fascinating", "flawless", "glamorous", "glorious", "illumination", "impressive", "improved",
"inspirational", "magnificent", "majestic", "hyperrealistic", "smooth", "sharp", "focus", "stunning", "detailed",
"intricate", "dramatic", "high", "quality", "perfect", "light", "ultra", "highly", "radiant", "satisfying",
"soothing", "sophisticated", "stylish", "sublime", "terrific", "touching", "timeless", "wonderful", "unbelievable",
"elegant", "awesome", "amazing", "dynamic", "trendy",
]
```

قد تلاحظ في قائمة `words`، أن هناك كلمات معينة يمكن اقترانها لخلق شيء أكثر معنى. على سبيل المثال، يمكن دمج كلمتي "high" و "quality" لتصبح "high quality". دعونا نقرن هذه الكلمات معًا ونزيل الكلمات التي لا يمكن اقترانها.

```py
word_pairs = ["highly detailed", "high quality", "enhanced quality", "perfect composition", "dynamic light"]

def find_and_order_pairs(s, pairs):
words = s.split()
found_pairs = []
for pair in pairs:
pair_words = pair.split()
if pair_words[0] in words and pair_words[1] in words:
found_pairs.append(pair)
words.remove(pair_words[0])
words.remove(pair_words[1])

for word in words[:]:
for pair in pairs:
if word in pair.split():
words.remove(word)
break
ordered_pairs = ", ".join(found_pairs)
remaining_s = ", ".join(words)
return ordered_pairs, remaining_s
```

بعد ذلك، قم بتنفيذ فئة [`~transformers.LogitsProcessor` مخصصة] تقوم بتعيين الرموز في قائمة `words` بقيمة 0 وتعيين الرموز غير الموجودة في قائمة `words` إلى قيمة سالبة حتى لا يتم اختيارها أثناء التوليد. بهذه الطريقة، يكون التوليد متحيزًا نحو الكلمات الموجودة في قائمة `words`. بعد استخدام كلمة من القائمة، يتم أيضًا تعيينها إلى قيمة سالبة حتى لا يتم اختيارها مرة أخرى.

```py
class CustomLogitsProcessor(LogitsProcessor):
def __init__(self, bias):
super().__init__()
self.bias = bias

def __call__(self, input_ids, scores):
if len(input_ids.shape) == 2:
last_token_id = input_ids[0, -1]
self.bias[last_token_id] = -1e10
return scores + self.bias

word_ids = [tokenizer.encode(word, add_prefix_space=True)[0] for word in words]
bias = torch.full((tokenizer.vocab_size,), -float("Inf")).to("cuda")
bias[word_ids] = 0
processor = CustomLogitsProcessor(bias)
processor_list = LogitsProcessorList([processor])
```

قم بدمج الفاصل الزمني وأسلوب "cinematic" المحدد في قاموس `styles` سابقًا.

```py
prompt = "a cat basking in the sun on a roof in Turkey"
style = "cinematic"

prompt = styles[style].format(prompt=prompt)
prompt
"cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain"
```

قم بتحميل برنامج تشفير GPT2 ونموذج من نقطة التحقق [Gustavosta/MagicPrompt-Stable-Diffusion](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion) (تم تدريب نقطة التحقق هذه خصيصًا لتوليد الفواصل الزمنية) لتحسين الفاصل الزمني.

```py
tokenizer = GPT2Tokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
model = GPT2LMHeadModel.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion", torch_dtype=torch.float16).to(
"cuda"
)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
token_count = inputs["input_ids"].shape[1]
max_new_tokens = 50 - token_count

generation_config = GenerationConfig(
penalty_alpha=0.7,
top_k=50,
eos_token_id=model.config.eos_token_id،
pad_token_id=model.config.eos_token_id،
pad_token=model.config.pad_token_id،
do_sample=True,
)

with torch.no_grad():
generated_ids = model.generate(
input_ids=inputs["input_ids"],
attention_mask=inputs["attention_mask"],
max_new_tokens=max_new_tokens,
generation_config=generation_config,
logits_processor=proccesor_list,
)
```

بعد ذلك، يمكنك دمج الفاصل الزمني المدخلات والفاصل الزمني المولد. لا تتردد في إلقاء نظرة على ما هو الفاصل الزمني المولد (`generated_part`)، وأزواج الكلمات التي تم العثور عليها (`pairs`)، والكلمات المتبقية (`words`). كل هذا مضمن في `enhanced_prompt`.

```py
output_tokens = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]
input_part, generated_part = output_tokens[0][: len(prompt)], output_tokens[0][len(prompt) :]
pairs, words = find_and_order_pairs(generated_part, word_pairs)
formatted_generated_part = pairs + ", " + words
enhanced_prompt = input_part + ", " + formatted_generated_part
enhanced_prompt
["cinematic film still of a cat basking in the sun on a roof in Turkey, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain quality sharp focus beautiful detailed intricate stunning amazing epic"]
```

أخيرًا، قم بتحميل خط أنابيب وLORA ضوضاء التعويض بوزن *منخفض* لتوليد صورة باستخدام الفاصل الزمني المحسن.

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
"RunDiffusion/Juggernaut-XL-v9"، torch_dtype=torch.float16، variant="fp16"
).to("cuda")

pipeline.load_lora_weights(
"stabilityai/stable-diffusion-xl-base-1.0"،
weight_name="sd_xl_offset_example-lora_1.0.safetensors"،
adapter_name="offset"،
)
pipeline.set_adapters(["offset"], adapter_weights=[0.2])

image = pipeline(
enhanced_prompt,
width=1152,
height=896,
guidance_scale=7.5,
num_inference_steps=25,
).images[0]
image
```

فيما يلي مثال على صورة تم إنشاؤها باستخدام الفاصل الزمني المحسن:

![صورة لقطة تستمتع بأشعة الشمس على سطح في تركيا، مع تفاصيل دقيقة وميزانية إنتاج عالية وجودة صورة سينمائية](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/enhanced-prompt.png)

"لقطة فيلم سينمائي لقطة لقطة تستمتع بأشعة الشمس على سطح في تركيا، مفصلة للغاية، وميزانية إنتاج عالية، وفيلم هوليوود، وشاشة سينمائية، ومزاجية، وملحمية، وجميلة، وحبيبات فيلم حادة التركيز جميلة مفصلة معقدة مذهلة ملحمية".
بالتأكيد، سأقوم بترجمة النص الموجود في الفقرات والعناوين مع الالتزام بالتعليمات التي قدمتها.

## وزن المطالبة

يوفر وزن المطالبة طريقة لتأكيد أو تقليل أهمية أجزاء معينة من المطالبة، مما يتيح مزيدًا من التحكم في الصورة المولدة. يمكن أن تتضمن المطالبة عدة مفاهيم، والتي يتم تحويلها إلى تضمينات نصية سياقية. تستخدم النماذج التضمينات لتكييف طبقات الاهتمام المتقاطع الخاصة بها لتوليد صورة. (اقرأ المنشور على مدونة Stable Diffusion لمعرفة المزيد حول كيفية عملها).

يعمل وزن المطالبة عن طريق زيادة أو تقليل مقياس متجه التضمين النصي الذي يقابله في المطالبة. قد لا ترغب في أن يركز النموذج على جميع المفاهيم بالتساوي. أسهل طريقة لإعداد التضمينات ذات الأوزان المطالبة هي استخدام Compel، وهي مكتبة لوزن المطالبة النصية ودمجها. بمجرد الحصول على التضمينات ذات الأوزان المطالبة، يمكن تمريرها إلى أي خط أنابيب يحتوي على معلمة prompt_embeds (واختياريًا معلمة negative_prompt_embeds)، مثل StableDiffusionPipeline وStableDiffusionControlNetPipeline وStableDiffusionXLPipeline.

هذا الدليل سوف يريك كيفية وزن ودمج مطالباتك مع Compel في 🤗 Diffusers.

قبل أن تبدأ، تأكد من تثبيت أحدث إصدار من Compel:

للإرشادات في هذا القسم، سنقوم بتوليد صورة باستخدام المطالبة "قطة حمراء تلعب بكرة" باستخدام StableDiffusionPipeline:

### الوزن

ستلاحظ أنه لا توجد "كرة" في الصورة! دعنا نستخدم Compel لزيادة وزن مفهوم "الكرة" في المطالبة. قم بإنشاء كائن Compel، ومرر محدد مواقع ومعالج نصي إليه:

يستخدم Compel + أو - لزيادة أو تقليل وزن كلمة في المطالبة. لزيادة وزن "الكرة":

يمكنك زيادة أو تقليل وزن عدة مفاهيم في نفس المطالبة:

### المزج

يمكنك أيضًا إنشاء مزيج مرجح من المطالبات عن طريق إضافة .blend () إلى قائمة من المطالبات وتمرير بعض الأوزان. قد لا ينتج مزيجك دائمًا النتيجة التي تتوقعها لأنه يكسر بعض الافتراضات حول كيفية عمل المشفر النصي، لذا قم بالتجربة والاستمتاع به!

### العطف

يعمل العطف على نشر كل مطالبة بشكل مستقل ويقوم بدمج نتائجها بواسطة مجموعها المرجح. أضف .and () في نهاية قائمة من المطالبات لإنشاء اقتران:

### الانقلاب النصي

الانقلاب النصي هو تقنية لتعلم مفهوم محدد من بعض الصور التي يمكنك استخدامها لتوليد صور جديدة مشروطة بذلك المفهوم.

قم بإنشاء خط أنابيب واستخدم وظيفة ~loaders.TextualInversionLoaderMixin.load_textual_inversion لتحميل التضمينات الانقلابية النصية (لا تتردد في تصفح Stable Diffusion Conceptualizer لمفاهيم مدربة 100+):

يوفر Compel فئة DiffusersTextualInversionManager لتبسيط وزن المطالبة مع الانقلاب النصي. قم بتنفيذ DiffusersTextualInversionManager ومرره إلى فئة Compel:

قم بدمج المفهوم لشرط مطالبة باستخدام بناء الجملة <concept>:

### DreamBooth

DreamBooth هي تقنية لتوليد صور سياقية لموضوع ما بناءً على عدد قليل فقط من صور الموضوع للتدريب عليها. إنه مشابه للانقلاب النصي، ولكن DreamBooth يقوم بتدريب النموذج بالكامل في حين أن الانقلاب النصي يقوم فقط بتدريب التضمينات النصية. وهذا يعني أنه يجب استخدام ~DiffusionPipeline.from_pretrained لتحميل نموذج DreamBooth (لا تتردد في تصفح مكتبة Stable Diffusion Dreambooth Concepts لمفاهيم مدربة 100+):

قم بإنشاء فئة Compel مع محدد مواقع ومعالج نصي، ومرر مطالبتك إليها. اعتمادًا على النموذج الذي تستخدمه، ستحتاج إلى تضمين المعرف الفريد للنموذج في مطالبتك. على سبيل المثال، يستخدم نموذج "dndcoverart-v1" المعرف "dndcoverart":
بالتأكيد، سألتزم بالتعليمات المذكورة.

### Stable Diffusion XL

يتميز Stable Diffusion XL (SDXL) بوجود رمزين ومشفرين نصيين، لذا فإن استخدامه يختلف قليلًا. ولمعالجة ذلك، يجب تمرير كلا الرمزين والمشفرين إلى فئة Compel:

هذه المرة، دعنا نزيد وزن "ball" بمعامل 1.5 للملمح الأول، ونقلل وزن "ball" إلى 0.6 للملمح الثاني. تتطلب أيضًا [StableDiffusionXLPipeline] [pooled_prompt_embeds] (واختياريًا [negative_pooled_prompt_embeds])، لذا يجب تمرير هذه العناصر إلى الأنبوب جنبًا إلى جنب مع موترات التكييف:

# تطبيق الأوزان
الملمح = ["قط أحمر يلعب بكرة (1.5)"، "قط أحمر يلعب بكرة (0.6)"]
التكييف، المجمع = compel(prompt)

# إنشاء الصورة
المولد = [torch.Generator().manual_seed(33) لـ _ في نطاق (طول (الملمح))]
الصور = pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, num_inference_steps=30).images
make_image_grid(images, rows=1, cols=2)