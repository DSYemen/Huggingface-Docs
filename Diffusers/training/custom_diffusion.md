# Custom Diffusion
تقنية Custom Diffusion هي تقنية تدريب لشخصنة نماذج توليد الصور. مثل الانعكاس النصي (Textual Inversion) وDreamBooth وLoRA، تتطلب تقنية Custom Diffusion أيضًا عددًا قليلاً فقط من الصور (حوالي 4-5 صور) كمثال. تعمل هذه التقنية من خلال تدريب الأوزان في طبقات الاهتمام المتقاطع فقط، وتستخدم كلمة خاصة لتمثيل المفهوم الذي تم تعلمه حديثًا. ما يميز Custom Diffusion هو قدرتها أيضًا على تعلم مفاهيم متعددة في نفس الوقت.

إذا كنت تقوم بالتدريب على وحدة معالجة رسومية (GPU) ذات ذاكرة وصول عشوائي (VRAM) محدودة، فيجب عليك تجربة تمكين xFormers باستخدام --enable_xformers_memory_efficient_attention لتسريع التدريب مع تقليل متطلبات ذاكرة VRAM (16 جيجابايت). لتوفير المزيد من الذاكرة، أضف --set_grads_to_none في حجة التدريب لتعيين التدرجات إلى None بدلاً من الصفر (قد يسبب هذا الخيار بعض المشكلات، لذا إذا واجهت أي مشاكل، حاول إزالة هذا المعامل).

سيتناول هذا الدليل بالتفصيل نص البرنامج النصي train_custom_diffusion.py لمساعدتك على التعرف عليه بشكل أفضل، وكيف يمكنك تكييفه مع حالتك الاستخدام الخاصة.

قبل تشغيل البرنامج النصي، تأكد من تثبيت المكتبة من المصدر:

انتقل إلى المجلد الذي يحتوي على البرنامج النصي للتدريب وقم بتثبيت التبعيات المطلوبة:

🤗 Accelerate هي مكتبة تساعدك على التدريب على وحدات معالجة رسومية/وحدات معالجة رسومات متعددة أو مع دقة مختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على أجهزتك وبيئتك. اطلع على الجولة السريعة في 🤗 Accelerate لمعرفة المزيد.

قم بتهيئة بيئة 🤗 Accelerate:

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

أو إذا لم يدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام:

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل إنشاء مجموعة بيانات للتدريب لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع البرنامج النصي للتدريب.

تسلط الأقسام التالية الضوء على أجزاء من البرنامج النصي للتدريب والتي تعد مهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب البرنامج النصي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة البرنامج النصي واطلعنا على أي أسئلة أو مخاوف.

## معلمات البرنامج النصي
يحتوي البرنامج النصي للتدريب على جميع المعلمات لمساعدتك على تخصيص عملية التدريب الخاصة بك. يمكن العثور عليها في دالة parse_args(). تأتي الدالة مع القيم الافتراضية، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا أردت.

على سبيل المثال، لتغيير دقة صورة الإدخال:

وصف العديد من المعلمات الأساسية في دليل تدريب DreamBooth، لذلك يركز هذا الدليل على المعلمات الفريدة لتقنية Custom Diffusion:

- --freeze_model: يقوم بتجميد المعلمات الرئيسية والقيمية في طبقة الاهتمام المتقاطع؛ الافتراضي هو crossattn_kv، ولكن يمكنك تعيينه على crossattn لتدريب جميع المعلمات في طبقة الاهتمام المتقاطع
- --concepts_list: لتعلم مفاهيم متعددة، قم بتوفير مسار إلى ملف JSON يحتوي على المفاهيم
- --modifier_token: كلمة خاصة تستخدم لتمثيل المفهوم الذي تم تعلمه
- --initializer_token: كلمة خاصة تستخدم لتهيئة تضمينات modifier_token

### خسارة الحفاظ على الأولوية
خسارة الحفاظ على الأولوية هي طريقة تستخدم عينات مولدة من النموذج نفسه لمساعدته على تعلم كيفية إنشاء صور أكثر تنوعًا. نظرًا لأن صور العينات المولدة هذه تنتمي إلى نفس الفئة التي قدمتها، فإنها تساعد النموذج على الاحتفاظ بما تعلمه حول الفئة وكيف يمكنه استخدام ما يعرفه بالفعل عن الفئة لإنشاء تكوينات جديدة.

وصف العديد من معلمات خسارة الحفاظ على الأولوية في دليل تدريب DreamBooth.

### الضبط
تتضمن تقنية Custom Diffusion تدريب الصور المستهدفة باستخدام مجموعة صغيرة من الصور الحقيقية لمنع الإفراط في التكيّف. كما يمكنك أن تتخيل، يمكن أن يكون من السهل القيام بذلك عندما تقوم بالتدريب على عدد قليل من الصور فقط! قم بتنزيل 200 صورة حقيقية باستخدام clip_retrieval. يجب أن يكون class_prompt من نفس فئة الصور المستهدفة. يتم تخزين هذه الصور في class_data_dir.

لتمكين الضبط، أضف المعلمات التالية:

- --with_prior_preservation: ما إذا كان سيتم استخدام خسارة الحفاظ على الأولوية
- --prior_loss_weight: يتحكم في تأثير خسارة الحفاظ على الأولوية على النموذج
- --real_prior: ما إذا كان سيتم استخدام مجموعة صغيرة من الصور الحقيقية لمنع الإفراط في التكيّف

آمل أن تكون الترجمة واضحة ومفهومة، لا تتردد في إخباري إذا كانت هناك أي نقاط تحتاج إلى توضيح أو إذا كنت تريد مني اتباع أي تعليمات إضافية.
## نص البرنامج النصي للتدريب

يحتوي نص البرنامج النصي للتدريب على Custom Diffusion على فئتين من مجموعات البيانات:

- [`CustomDiffusionDataset`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py#L165): تقوم بمعالجة الصور، وصور الفئات، والمحفزات لأغراض التدريب

- [`PromptDataset`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py#L148): تحضير المحفزات لتوليد صور الفئات

بعد ذلك، يتم [إضافة `modifier_token` إلى المحلل](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py#L811)، وتحويلها إلى معرفات الرموز، ويتم تغيير حجم تضمين الرموز لمراعاة `modifier_token` الجديد. ثم يتم تهيئة تضمينات `modifier_token` باستخدام تضمينات `initializer_token`. يتم تجميد جميع المعلمات في encoder النصي، باستثناء تضمينات الرموز لأن هذا ما يحاول النموذج تعلم ربطه بالمفاهيم.

```py
params_to_freeze = itertools.chain(
text_encoder.text_model.encoder.parameters(),
text_encoder.text_model.final_layer_norm.parameters(),
text_encoder.text_model.embeddings.position_embedding.parameters(),
)
freeze_params(params_to_freeze)
```

الآن، ستحتاج إلى إضافة [أوزان Custom Diffusion](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py#L911C3-L911C3) إلى طبقات الاهتمام. تعد هذه الخطوة مهمة جدًا للحصول على الشكل والحجم الصحيح لأوزان الاهتمام، ولتعيين العدد المناسب من معالجات الاهتمام في كل كتلة UNet.

```py
st = unet.state_dict()
for name, _ in unet.attn_processors.items():
cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
if name.startswith("mid_block"):
hidden_size = unet.config.block_out_channels[-1]
elif name.startswith("up_blocks"):
block_id = int(name[len("up_blocks.")])
hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
elif name.startswith("down_blocks"):
block_Multiplier = int(name[len("down_blocks.")])
hidden_size = unet.config.block_out_channels[block_Multiplier]
layer_name = name.split(".processor")[0]
weights = {
"to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
"to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
}
if train_q_out:
weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
if cross_attention_dim is not None:
custom_diffusion_attn_procs[name] = attention_class(
train_kv=train_kv,
train_q_out=train_q_out,
hidden_size=hidden_size,
cross_attention_dim=cross_attention_dim,
).to(unet.device)
custom_diffusion_attn_procs[name].load_state_dict(weights)
else:
custom_diffusion_attn_procs[name] = attention_class(
train_kv=False,
train_q_out=False,
hidden_size=hidden_size,
cross_attention_dim=cross_attention_dim,
)
del st
unet.set_attn_processor(custom_diffusion_attn_procs)
custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)
```

يتم تهيئة [المحسن](https://github.com/huggingface/diffusers/blob/84cd9e8d01adb47f046b1ee449fc76a0c32dc4e2/examples/custom_diffusion/train_custom_diffusion.py#L982) لتحديث معلمات طبقة الاهتمام المتقاطع:

```py
optimizer = optimizer_class(
itertools.chain(text_encoder.get_input_embeddings().parameters(), custom_diffusion_layers.parameters())
if args.modifier_token is not None
else custom_diffusion_layers.parameters(),
lr=args.learning_rate,
betas=(args.adam_beta1, args.adam_beta2),
weight_decay=args.adam_weight_decay,
eps=args.adam_epsilon,
)
```

في [حلقة التدريب](https://github.com/huggingface/diffusers/blob/84cd9e8d01adb47f046b1ee449fc76a0c32dc4e2/examples/custom_diffusion/train_custom_diffusion.py#L1048)، من المهم تحديث التضمينات للمفهوم الذي تحاول تعلمه فقط. وهذا يعني تعيين تدرجات جميع تضمينات الرموز الأخرى إلى الصفر:

```py
if args.modifier_token is not None:
if accelerator.num_processes > 1:
grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
else:
grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
for i in range(len(modifier_token_id[1:])):
index_grads_to_zero = index_grads_to_zero & (
torch.arange(len(tokenizer)) != modifier_token_id[i]
)
grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
index_grads_to_zero, :
].fill_(0)
```

## إطلاق البرنامج النصي

بمجرد إجراء جميع التغييرات أو إذا كنت راضيًا عن التكوين الافتراضي، فأنت مستعد لإطلاق برنامج التدريب النصي! 🚀

في هذا الدليل، ستقوم بتنزيل واستخدام هذه الصور [صور القطط](https://www.cs.cmu.edu/~custom-diffusion/assets/data.zip) كمثال. يمكنك أيضًا إنشاء واستخدام مجموعة البيانات الخاصة بك إذا أردت (راجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset)).

قم بتعيين متغير البيئة `MODEL_NAME` إلى معرف نموذج على Hub أو مسار إلى نموذج محلي، و`INSTANCE_DIR` إلى المسار الذي قمت بتنزيل صور القطط إليه، و`OUTPUT_DIR` إلى المكان الذي تريد حفظ النموذج فيه. ستستخدم `<new1>` ككلمة خاصة لربط التضمينات الجديدة التي تم تعلمها. يقوم البرنامج النصي بإنشاء وحفظ نقاط تحقق النموذج وملف pytorch_custom_diffusion_weights.bin إلى مستودعك.

لمراقبة تقدم التدريب باستخدام Weights and Biases، أضف المعلمة `--report_to=wandb` إلى أمر التدريب وحدد موجه التحقق من الصحة باستخدام `--validation_prompt`. هذا مفيد للتصحيح وحفظ النتائج الوسيطة.

<Tip>

إذا كنت تتدرب على وجوه بشرية، فقد وجد فريق Custom Diffusion أن المعلمات التالية تعمل بشكل جيد:

- `--learning_rate=5e-6`

- يمكن أن يكون `--max_train_steps` أي رقم بين 1000 و2000

- `--freeze_model=crossattn`

- استخدم ما لا يقل عن 15-20 صورة للتدريب

</Tip>

<hfoptions id="training-inference">
<hfoption id="single concept">

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="./data/cat"

accelerate launch train_custom_diffusion.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--class_data_dir=./real_reg/samples_cat/ \
--with_prior_preservation \
--real_prior \
--prior_loss_weight=1.0 \
--class_prompt="cat" \
--num_class_images=200 \
--instance_prompt="photo of a <new1> cat" \
--resolution=512 \
--train_batch_size=2 \
--learning_rate=1e-5 \
--lr_warmup_steps=0 \
--max_train_steps=250 \
--scale_lr \
--hflip \
--modifier_token "<new1>" \
--validation_prompt="<new1> cat sitting in a bucket" \
--report_to="wandb" \
--push_to_hub
```

</hfoption>
<hfoption id="multiple concepts">

يمكن لـ Custom Diffusion أيضًا تعلم مفاهيم متعددة إذا قدمت ملف [JSON](https://github.com/adobe-research/custom-diffusion/blob/main/assets/concept_list.json) مع بعض التفاصيل حول كل مفهوم يجب تعلمه.

قم بتشغيل استرداد CLIP لجمع بعض الصور الحقيقية لاستخدامها في التنظيم:

```bash
pip install clip-retrieval
python retrieve.py --class_prompt {} --class_data_dir {} --num_class_images 200
```

بعد ذلك، يمكنك إطلاق البرنامج النصي:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_custom_diffusion.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--output_dir=$OUTPUT_DIR \
--concepts_list=./concept_list.json \
--with_prior_preservation \
--real_prior \
--prior_loss_weight=1.0 \
--resolution=512 \
--train_batch_size=2 \
--learning_rate=1e-5 \
--lr_warmup_steps=0 \
--max_train_steps=500 \
--num_class_images=200 \
--scale_lr \
--hflip \
--modifier_token "<new1>+<new2>" \
--push_to_hub
```

</hfoption>
</hfoptions>

بمجرد الانتهاء من التدريب، يمكنك استخدام نموذج Custom Diffusion الخاص بك للاستنتاج.

<hfoptions id="training-inference">
<hfoption id="single concept">

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
).to("cuda")
pipeline.unet.load_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
pipeline.load_textual_inversion("path-to-save-model", weight_name="<new1>.bin")

image = pipeline(
"<new1> cat sitting in a bucket",
num_inference_steps=100,
guidance_scale=6.0,
eta=1.0,
).images[0]
image.save("cat.png")
```

</hfoption>
<hfoption id="multiple concepts">

```py
import torch
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("sayakpaul/custom-diffusion-cat-wooden-pot", torch_dtype=torch.float16).to("cuda")
pipeline.unet.load_attn_procs(model_id, weight_name="pytorch_custom_diffusion_weights.bin")
pipeline.load_textual_inversion(model_id, weight_name="<new1>.bin")
pipeline.load_textual_inversion(model_id, weight_name="<new2>.bin")

image = pipeline(
"the <new1> cat sculpture in the style of a <new2> wooden pot",
num_inference_steps=100,
guidance_scale=6.0,
eta=1.0,
).images[0]
image.save("multi-subject.png")
```

</hfoption>
</hfoptions>

## الخطوات التالية

تهانينا على تدريب نموذج باستخدام Custom Diffusion! 🎉 لمزيد من المعلومات:

- اقرأ منشور المدونة [تخصيص المفهوم المتعدد لنشر النص إلى الصورة](https://www.cs.cmu.edu/~custom-diffusion/) لمعرفة المزيد من التفاصيل حول النتائج التجريبية من فريق Custom Diffusion.