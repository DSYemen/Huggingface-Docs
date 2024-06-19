## الانقلاب النصي

تقنية تدريب لشخصنة نماذج توليد الصور باستخدام عدد قليل فقط من صور الأمثلة لما تريد أن تتعلمه. تعمل هذه التقنية من خلال تعلم وتحديث تضمين النص (ترتبط التضمينات الجديدة بكلمة خاصة يجب استخدامها في الفكرة) لمطابقة صور الأمثلة التي توفرها.

إذا كنت تتدرب على وحدة معالجة رسومات (GPU) ذات ذاكرة وصول عشوائي (VRAM) محدودة، فيجب عليك تجربة تمكين المعلمات `gradient_checkpointing` و`mixed_precision` في أمر التدريب. يمكنك أيضًا تقليل البصمة الخاصة بك باستخدام اهتمام فعال للذاكرة مع [xFormers](../optimization/xformers). يتم أيضًا دعم التدريب JAX/Flax للتدريب الفعال على وحدات معالجة الرسومات (TPUs) ووحدات معالجة الرسومات (GPUs)، ولكنه لا يدعم نقطة تفتيش التدرج أو xFormers. باستخدام نفس التكوين والإعداد مثل PyTorch، يجب أن يكون نص Flax أسرع بنسبة 70% على الأقل!

سيتناول هذا الدليل النص البرمجي [textual_inversion.py](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py) لمساعدتك على التعرف عليه بشكل أفضل، وكيف يمكنك تكييفه مع حالتك الاستخدام.

قبل تشغيل النص، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

انتقل إلى مجلد المثال باستخدام نص التدريب وقم بتثبيت التبعيات المطلوبة للنص الذي تستخدمه:

<hfoptions id="installation">
<hfoption id="PyTorch">

```bash
cd examples/textual_inversion
pip install -r requirements.txt
```

</hfoption>
<hfoption id="Flax">

```bash
cd examples/textual_inversion
pip install -r requirements_flax.txt
```

</hfoption>
</hfoptions>

🤗 Accelerate هي مكتبة للمساعدة في التدريب على وحدات معالجة الرسومات (GPU) / وحدات معالجة الرسومات (TPU) متعددة أو مع الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على الأجهزة والبيئة الخاصة بك. الق نظرة على جولة سريعة في 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.

قم بتهيئة بيئة 🤗 Accelerate:

```bash
accelerate config
```

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

```bash
accelerate config default
```

أو إذا لم يدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع نص التدريب.

تسلط الأقسام التالية الضوء على أجزاء من نص التدريب المهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب النص بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة النص [script](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py) ودعنا نعرف إذا كان لديك أي أسئلة أو مخاوف.

## معلمات النص

يحتوي نص التدريب على العديد من المعلمات لمساعدتك في تخصيص تشغيل التدريب وفقًا لاحتياجاتك. يتم سرد جميع المعلمات ووصفها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/839c2a5ece0af4e75530cb520d77bc7ed8acf474/examples/textual_inversion/textual_inversion.py#L176). حيثما ينطبق، توفر Diffusers القيم الافتراضية لكل معلمة مثل حجم دفعة التدريب ومعدل التعلم، ولكن لا تتردد في تغيير هذه القيم في أمر التدريب إذا كنت ترغب في ذلك.

على سبيل المثال، لزيادة عدد خطوات تراكم التدرجات فوق القيمة الافتراضية 1:

```bash
accelerate launch textual_inversion.py \
--gradient_accumulation_steps=4
```

بعض المعلمات الأساسية والمهمة الأخرى التي يجب تحديدها تشمل:

- `--pretrained_model_name_or_path`: اسم النموذج على Hub أو مسار محلي للنموذج المُدرب مسبقًا
- `--train_data_dir`: المسار إلى مجلد يحتوي على مجموعة بيانات التدريب (صور الأمثلة)
- `--output_dir`: المكان الذي سيتم فيه حفظ النموذج المدرب
- `--push_to_hub`: ما إذا كان سيتم دفع النموذج المدرب إلى Hub
- `--checkpointing_steps`: تكرار حفظ نقطة تفتيش أثناء تدريب النموذج؛ هذا مفيد إذا تم مقاطعة التدريب لسبب ما، فيمكنك الاستمرار في التدريب من تلك النقطة عن طريق إضافة `--resume_from_checkpoint` إلى أمر التدريب
- `--num_vectors`: عدد المتجهات لتعلم التضمينات بها؛ زيادة هذا المعلمة تساعد النموذج على التعلم بشكل أفضل ولكنها تأتي بتكاليف تدريب متزايدة
- `--placeholder_token`: الكلمة الخاصة لربط التضمينات المكتسبة (يجب استخدام الكلمة في فكرتك للاستدلال)
- `--initializer_token`: كلمة واحدة تصف بشكل عام الكائن أو الأسلوب الذي تحاول التدريب عليه
- `--learnable_property`: ما إذا كنت تدرب النموذج لتعلم "أسلوب" جديد (على سبيل المثال، أسلوب الرسم لفان جوخ) أو "كائن" (على سبيل المثال، كلبك)

## نص التدريب

على عكس بعض نصوص التدريب الأخرى، يحتوي نص textual_inversion.py على فئة مجموعة بيانات مخصصة، [`TextualInversionDataset`](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L487) لإنشاء مجموعة بيانات. يمكنك تخصيص حجم الصورة، والرمز النائب، وطريقة الاستيفاء، وما إذا كان سيتم اقتصاص الصورة، والمزيد. إذا كنت بحاجة إلى تغيير طريقة إنشاء مجموعة البيانات، فيمكنك تعديل `TextualInversionDataset`.

بعد ذلك، ستجد رمز معالجة مجموعة البيانات وحلقة التدريب في دالة [`main()`](https://github.com/huggingface/diffusers/blob/839c2a5ece0af4e75530cb520d77bc7ed8acf474/examples/textual_inversion/textual_inversion.py#L573).

يبدأ النص بتحميل [الرموز](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L616)، [المخطط والنماذج](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L622):

```py
# تحميل الرموز
if args.tokenizer_name:
tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
elif args.pretrained_model_name_or_path:
tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

# تحميل المخطط والنماذج
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(
args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
unet = UNet2DConditionModel.from_pretrained(
args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
)
```

يتم إضافة الرمز النائب الخاص [بالرموز](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L632) بعد ذلك، ويتم إعادة ضبط التضمين للتعويض عن الرمز الجديد.

بعد ذلك، يقوم النص [بإنشاء مجموعة بيانات](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L716) من `TextualInversionDataset`:

```py
train_dataset = TextualInversionDataset(
data_root=args.train_data_dir,
tokenizer=tokenizer،
size=args.resolution,
placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
repeats=args.repeats,
learnable_property=args.learnable_property,
center_crop=args.center_crop,
set="train"،
)
train_dataloader = torch.utils.data.DataLoader(
train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
)
```

أخيرًا، تتولى حلقة [التدريب](https://github.com/huggingface/diffusers/blob/b81c69e489aad3a0ba73798c459a33990dc4379c/examples/textual_inversion/textual_inversion.py#L784) كل شيء آخر بدءًا من التنبؤ ببقايا الضوضاء وحتى تحديث أوزان التضمين للرمز النائب الخاص.

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة التشويش.
## تشغيل السكربت

عندما تنتهي من إجراء جميع التغييرات أو تكون راضيًا عن التكوين الافتراضي، ستكون جاهزًا لتشغيل سكربت التدريب! 🚀

بالنسبة لهذا الدليل، ستقوم بتنزيل بعض الصور لـ [لعبة قط](https://huggingface.co/datasets/diffusers/cat_toy_example) وحفظها في دليل. ولكن تذكر، يمكنك إنشاء واستخدام مجموعة البيانات الخاصة بك إذا أردت (راجع الدليل [إنشاء مجموعة بيانات للتدريب](create_dataset)).

بعد التدريب، يمكنك استخدام النموذج الذي تدرب حديثًا للتنبؤ مثل:

تهانينا على تدريب نموذج الانعكاس النصي الخاص بك! 🎉 لمعرفة المزيد عن كيفية استخدام نموذجك الجديد، قد تكون الأدلة التالية مفيدة:

- تعرف على كيفية [تحميل انعكاسات نصية](../using-diffusers/loading_adapters) واستخدامها أيضًا كتعليقات سلبية.
- تعلم كيفية استخدام [الانعكاس النصي](textual_inversion_inference) للتنبؤ باستخدام Stable Diffusion 1/2 و Stable Diffusion XL.