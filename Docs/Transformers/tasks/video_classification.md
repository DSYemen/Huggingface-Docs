## تصنيف الفيديو

يعد تصنيف الفيديو مهمة تتمثل في تعيين تسمية أو فئة لفيديو بأكمله. من المتوقع أن يكون لكل فيديو فئة واحدة فقط. تتخذ نماذج تصنيف الفيديو فيديو كمدخلات وتعيد تنبؤًا بالفئة التي ينتمي إليها الفيديو. يمكن استخدام هذه النماذج لتصنيف محتوى الفيديو. أحد التطبيقات الواقعية لتصنيف الفيديو هو التعرف على الإجراء/النشاط، وهو مفيد لتطبيقات اللياقة البدنية. كما أنه يساعد الأشخاص ضعاف البصر، خاصة عند التنقل.

سيوضح هذا الدليل كيفية:

1. ضبط نموذج VideoMAE بدقة على مجموعة فرعية من مجموعة بيانات UCF101.
2. استخدام نموذج ضبط دقة للاستنتاج.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install -q pytorchvideo transformers evaluate
```

ستستخدم PyTorchVideo (المسماة `pytorchvideo`) لمعالجة الفيديو وإعداده.

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات UCF101

ابدأ بتحميل مجموعة فرعية من مجموعة بيانات UCF-101. سيعطيك هذا فرصة للتجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from huggingface_hub import hf_hub_download

>>> hf_dataset_identifier = "sayakpaul/ucf101-subset"
>>> filename = "UCF101_subset.tar.gz"
>>> file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")
```

بعد تنزيل المجموعة الفرعية، تحتاج إلى استخراج الأرشيف المضغوط:

```py
>>> import tarfile

>>> with tarfile.open(file_path) as t:
...     t.extractall(".")
```

بشكل عام، يتم تنظيم مجموعة البيانات على النحو التالي:

```bash
UCF101_subset/
train/
BandMarching/
video_1.mp4
video_2.mp4
...
Archery
video_1.mp4
video_2.mp4
...
...
val/
BandMarching/
video_1.mp4
video_2.mp4
...
Archery
video_1.mp4
video_2.mp4
...
...
test/
BandMarching/
video_1.mp4
video_2.mp4
...
Archery
video_1.mp4
video_2.mp4
...
...
```

بعد ذلك، يمكنك حساب عدد مقاطع الفيديو الإجمالية.

```py
>>> import pathlib
>>> dataset_root_path = "UCF101_subset"
>>> dataset_root_path = pathlib.Path(dataset_root_path)
```

```py
>>> video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
>>> video_count_val = len(list(dataset_root_
path.glob("val/*/*.avi")))
>>> video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
>>> video_total = video_count_train + video_count_val + video_count_test
>>> print(f"Total videos: {video_total}")
```

```py
>>> all_video_file_paths = (
...     list(dataset_root_path.glob("train/*/*.avi"))
...     + list(dataset_root_path.glob("val/*/*.avi"))
...     + list(dataset_root_path.glob("test/*/*.avi"))
... )
>>> all_video_file_paths[:5]
```

تظهر مسارات الفيديو (المفرزة) على النحو التالي:

```bash
...
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c04.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c06.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c02.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c06.avi'
...
```

ستلاحظ أن هناك مقاطع فيديو تنتمي إلى نفس المجموعة/المشهد حيث يتم تمثيل المجموعة بواسطة "g" في مسارات ملفات الفيديو. على سبيل المثال، "v_ApplyEyeMakeup_g07_c04.avi" و"v_ApplyEyeMakeup_g07_c06.avi".

بالنسبة لعمليات التقسيم التحقق والتقسيم، لا تريد أن تكون لديك مقاطع فيديو من نفس المجموعة/المشهد لمنع تسرب البيانات. تأخذ المجموعة الفرعية التي تستخدمها في هذا البرنامج التعليمي هذه المعلومات في الاعتبار.

بعد ذلك، ستقوم باستنتاج مجموعة العلامات الموجودة في مجموعة البيانات. كما ستقوم بإنشاء قاموسين سيكونان مفيدين عند تهيئة النموذج:

- `label2id`: يقوم بتعيين أسماء الفئات إلى أرقام صحيحة.
- `id2label`: يقوم بتعيين الأرقام الصحيحة إلى أسماء الفئات.

```py
>>> class_labels = sorted({str(path).split("/")[2] for path in all_video_file_paths})
>>> label2id = {label: i for i, label in enumerate(class_labels)}
>>> id2label = {i: label for label, i in label2id.items()}

>>> print(f"Unique classes: {list(label2id.keys())}.")

# فئات فريدة: ['ApplyEyeMakeup'، 'ApplyLipstick'، 'Archery'، 'BabyCrawling'، 'BalanceBeam'، 'BandMarching'، 'BaseballPitch'، 'Basketball'، 'BasketballDunk'، 'BenchPress'].
```

هناك 10 فئات فريدة. لكل فئة، هناك 30 مقطع فيديو في مجموعة التدريب.

## تحميل نموذج لضبط دقة

قم بتهيئة نموذج تصنيف فيديو من نقطة تفتيش مسبقة التدريب ومعالج الصور المرتبط بها. يأتي مشفر النموذج مع معلمات مسبقة التدريب، ويتم تهيئة رأس التصنيف بشكل عشوائي. سيكون معالج الصور مفيدًا عند كتابة خط أنابيب المعالجة المسبقة لمجموعة البيانات الخاصة بنا.

```py
>>> from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

>>> model_ckpt = "MCG-NJU/videomae-base"
>>> image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
>>> model = VideoMAEForVideoClassification.from_pretrained(
...     model_ckpt,
...     label2id=label2id,
...     id2label=id2label,
...     ignore_mismatched_sizes=True,  # قم بتوفير هذا في حالة كنت تخطط لضبط دقة نقطة تفتيش تمت معايرتها بالفعل
... )
```

بينما يتم تحميل النموذج، قد تلاحظ التحذير التالي:

```bash
تم تجاهل بعض أوزان نقطة تفتيش النموذج في MCG-NJU/videomae-base عند تهيئة VideoMAEForVideoClassification: [..., 'decoder.decoder_layers.1.attention.output.dense.bias'، 'decoder.decoder_layers.2.attention.attention.key.weight']
- هذا متوقع إذا كنت تقوم بتهيئة VideoMAEForVideoClassification من نقطة تفتيش لنموذج مدرب على مهمة أخرى أو بمعمارية أخرى (على سبيل المثال، تهيئة نموذج BertForSequenceClassification من نموذج BertForPreTraining).
- هذا غير متوقع إذا كنت تقوم بتهيئة VideoMAEForVideoClassification من نقطة تفتيش لنموذج تتوقع أن يكون متطابقًا تمامًا (تهيئة نموذج BertForSequenceClassification من نموذج BertForSequenceClassification).
لم يتم تهيئة بعض أوزان VideoMAEForVideoClassification من نقطة تفتيش النموذج في MCG-NJU/videomae-base وتم تهيئتها حديثًا: ['classifier.bias'، 'classifier.weight']
يجب عليك على الأرجح تدريب هذا النموذج على مهمة لأسفل لتتمكن من استخدامه للتنبؤات والاستدلال.
```

التحذير يخبرنا أننا نتخلص من بعض الأوزان (على سبيل المثال، وزن وتحيز طبقة "المصنف") ونقوم بتهيئة بعض الأوزان الأخرى بشكل عشوائي (وزن وتحيز طبقة "مصنف" جديدة). هذا متوقع في هذه الحالة، لأننا نقوم بإضافة رأس جديد لا تتوفر له أوزان مسبقة التدريب، لذا فإن المكتبة تحذرنا من أنه يجب علينا ضبط دقة النموذج قبل استخدامه للاستدلال، وهو ما سنقوم به بالضبط.

**لاحظ** أن [هذه النقطة](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) تؤدي إلى أداء أفضل في هذه المهمة لأن نقطة التفتيش تم الحصول عليها من خلال الضبط الدقيق لمهمة مماثلة ذات تداخل كبير في المجال. يمكنك التحقق من [هذه النقطة](https://huggingface.co/sayakpaul/videomae-base-finetuned-kinetics-finetuned-ucf101-subset) والتي تم الحصول عليها عن طريق الضبط الدقيق لـ `MCG-NJU/videomae-base-finetuned-kinetics`.
بالتأكيد، سأبدأ الترجمة من بعد التعليق الأول:

## إعداد مجموعات البيانات للتدريب

لمعالجة مقاطع الفيديو، ستستخدم مكتبة PyTorchVideo. ابدأ باستيراد الاعتمادات التي نحتاجها.

لتحويلات مجموعة بيانات التدريب، استخدم مزيجًا من الاستخراج الزمني الموحد، وتوحيد البكسل، والاقتصاص العشوائي، والقلب الأفقي العشوائي. بالنسبة لتحويلات مجموعة بيانات التحقق والتقييم، احتفظ بنفس سلسلة التحويل باستثناء الاقتصاص والقلب الأفقي. لمزيد من المعلومات حول تفاصيل هذه التحويلات، راجع الوثائق الرسمية لـ PyTorchVideo.

استخدم "image_processor" المرتبط بالنموذج المُدرب مسبقًا للحصول على المعلومات التالية:

- متوسط الانحراف المعياري للصورة والذي سيتم توحيد بكسلات إطار الفيديو معه.
- الدقة المكانية التي سيتم تغيير حجم إطارات الفيديو إليها.

ابدأ بتحديد بعض الثوابت.

الآن، قم بتعريف التحويلات المحددة لمجموعة البيانات ومجموعات البيانات على التوالي. بدءًا من مجموعة التدريب:

يمكن تطبيق نفس تسلسل سير العمل على مجموعات التحقق والتقييم:

**ملاحظة**: تم أخذ خطوط أنابيب مجموعة البيانات أعلاه من مثال PyTorchVideo الرسمي. نحن نستخدم الدالة "pytorchvideo.data.Ucf101()" لأنها مصممة لمجموعة بيانات UCF-101. في الواقع، فإنه يعيد كائن "pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset". تعد فئة "LabeledVideoDataset" الفئة الأساسية لجميع مقاطع الفيديو في مجموعة بيانات PyTorchVideo. لذلك، إذا كنت تريد استخدام مجموعة بيانات مخصصة غير مدعومة من PyTorchVideo، فيمكنك توسيع فئة "LabeledVideoDataset" وفقًا لذلك. راجع وثائق API للبيانات لمعرفة المزيد. أيضًا، إذا كانت مجموعة البيانات الخاصة بك تتبع بنية مشابهة (كما هو موضح أعلاه)، فإن استخدام "pytorchvideo.data.Ucf101()" يجب أن يعمل بشكل جيد.

يمكنك الوصول إلى وسيط "num_videos" لمعرفة عدد مقاطع الفيديو في مجموعة البيانات.

## تصور الفيديو المعالج مسبقًا للتصحيح الأفضل

```
يمكنك استيراد imageio وnumpy وIPython.display

قم بتعريف وظائف unnormalize_img وcreate_gif وdisplay_gif

الحصول على عينة من الفيديو من مجموعة بيانات التدريب

قم بتصور الفيديو المعالج باستخدام الدالة display_gif
```
بالتأكيد! فيما يلي ترجمة للنص الموجود في الفقرات والعناوين، مع اتباع التعليمات التي قدمتها:

## تدريب النموذج

يستخدم حزمة "ترانزفورمرز" (Transformers) من "هوجينج فيس" (Hugging Face) لتدريب النموذج. ولإنشاء مثيل من "ترينر" (Trainer)، تحتاج إلى تحديد تكوين التدريب ومقياس تقييم. والأهم من ذلك هو "ترينينج أرغومنتس" (TrainingArguments)، وهي فئة تحتوي على جميع السمات لتكوين التدريب. فهو يتطلب اسم مجلد الإخراج، والذي سيتم استخدامه لحفظ نقاط تفتيش النموذج. كما يساعد في مزامنة جميع المعلومات في مستودع النموذج على "هوب" (Hub).

معظم حجج التدريب واضحة، ولكن هناك واحدة مهمة هنا وهي 'remove_unused_columns=False'. سيؤدي هذا إلى إسقاط أي ميزات لا تستخدمها دالة استدعاء النموذج. افتراضيًا، يكون 'True' لأنه من المثالي عادةً إسقاط أعمدة الميزات غير المستخدمة، مما يجعل من السهل فك حزم المدخلات في دالة استدعاء النموذج. ولكن، في هذه الحالة، تحتاج إلى الميزات غير المستخدمة ('video' على وجه التحديد) من أجل إنشاء 'pixel_values' (وهو مفتاح إلزامي يتوقعه نموذجنا في مدخلاته).

يحتوي مجموعة البيانات التي يعيدها 'pytorchvideo.data.Ucf101()' لا تنفذ طريقة '__len__'. لذلك، يجب علينا تحديد 'max_steps' عند إنشاء مثيل 'TrainingArguments'.

بعد ذلك، تحتاج إلى تحديد دالة لحساب المقاييس من التوقعات، والتي ستستخدم 'metric' التي ستقوم بتحميلها الآن. المعالجة المسبقة الوحيدة التي يجب عليك القيام بها هي أخذ 'argmax' من logits المتوقعة:

**ملاحظة حول التقييم**:

في ورقة "فيديو ماي" (VideoMAE)، يستخدم المؤلفون استراتيجية التقييم التالية. حيث يقومون بتقييم النموذج على العديد من المقاطع من مقاطع الفيديو الاختبار وتطبيق محاصيل مختلفة على تلك المقاطع والإبلاغ عن النتيجة الإجمالية. ومع ذلك، من أجل البساطة والإيجاز، لا نأخذ ذلك في الاعتبار في هذا البرنامج التعليمي.

قم أيضًا بتعريف 'collate_fn'، والتي ستُستخدم لدمج الأمثلة في مجموعات. تتكون كل دفعة من مفتاحين، وهما 'pixel_values' و 'labels'.

بعد ذلك، قم فقط بتمرير كل هذا جنبًا إلى جنب مع مجموعات البيانات إلى 'Trainer':

قد تتساءل لماذا قمت بتمرير 'image_processor' كمحلل رموز عند معالجة البيانات بالفعل. هذا فقط للتأكد من أن ملف تكوين معالج الصور (المخزن بتنسيق JSON) سيتم تحميله أيضًا إلى المستودع على Hub.

الآن، نقوم بتعديل نموذجنا الدقيق عن طريق استدعاء طريقة 'train':

بمجرد اكتمال التدريب، شارك نموذجك على Hub باستخدام طريقة '~transformers.Trainer.push_to_hub' حتى يتمكن الجميع من استخدام نموذجك:

## الاستنتاج

الآن بعد أن قمت بتعديل نموذج، يمكنك استخدامه للاستنتاج!

قم بتحميل فيديو للاستنتاج:

أبسط طريقة لتجربة نموذجك المعدل للاستنتاج هي استخدامه في 'pipeline'. قم بتنفيذ عملية برمجة لنظام التصنيف بالفيديو باستخدام نموذجك، ومرر الفيديو إليه:

يمكنك أيضًا يدويًا تكرار نتائج 'pipeline' إذا أردت.

الآن، قم بتمرير المدخلات إلى النموذج وإرجاع 'logits':

فك تشفير 'logits'، نحصل على ما يلي:

آمل أن تكون الترجمة واضحة ومفهومة. لا تتردد في إخباري إذا كنت بحاجة إلى أي توضيحات أو إذا كانت هناك أي تعليمات إضافية.