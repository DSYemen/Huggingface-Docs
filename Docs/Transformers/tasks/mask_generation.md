# إنشاء الأقنعة

إن إنشاء القناع هو مهمة إنشاء أقنعة ذات معنى دلالي لصورة ما. هذه المهمة مشابهة جدًا لقطاع الصور، ولكن هناك العديد من الاختلافات. يتم تدريب نماذج تجزئة الصور على مجموعات بيانات موسومة ومقيدة بالتصنيفات التي شاهدتها أثناء التدريب؛ فهي تعيد مجموعة من الأقنعة والتصنيفات المقابلة، نظرًا للصورة.

يتم تدريب نماذج إنشاء القناع على كميات كبيرة من البيانات وتعمل في وضعين.

- وضع المطالبة: في هذا الوضع، يأخذ النموذج الصورة وطلبًا، حيث يمكن أن يكون الطلب موقعًا ثنائي الأبعاد (الإحداثيات XY) داخل كائن في الصورة أو مربعًا محددًا يحيط بكائن. في وضع المطالبة، يعيد النموذج القناع فقط على الكائن الذي يشير إليه الطلب.

- وضع تجزئة كل شيء: في تجزئة كل شيء، نظرًا للصورة، يقوم النموذج بتوليد كل قناع في الصورة. للقيام بذلك، يتم إنشاء شبكة من النقاط ووضعها فوق الصورة للاستدلال.

تتم دعم مهمة إنشاء القناع بواسطة نموذج تجزئة أي شيء (SAM). إنه نموذج قوي يتكون من محول رؤية قائم على محول، ومشفر مطالبة، وفك تشفير قناع محول ثنائي الاتجاه. يتم تشفير الصور والمطالبات، ويأخذ فك التشفير هذه التضمينات وينشئ أقنعة صالحة.

يعد SAM بمثابة نموذج أساسي قوي للتجزئة نظرًا لتغطيته الكبيرة للبيانات. يتم تدريبه على SA-1B، وهو مجموعة بيانات تحتوي على مليون صورة و1.1 مليار قناع.

في هذا الدليل، ستتعلم كيفية:

- الاستدلال في وضع تجزئة كل شيء مع المعالجة الدُفعية
- الاستدلال في وضع مطالبة النقطة
- الاستدلال في وضع مطالبة المربع

أولاً، دعنا نقوم بتثبيت المحولات:

الآن، بعد تثبيت المكتبة، يمكننا استيرادها واستخدامها:

## خط أنابيب إنشاء القناع

أسهل طريقة للاستدلال من نماذج إنشاء القناع هي استخدام خط أنابيب `mask-generation`.

دعنا نرى الصورة:

دعنا نقسم كل شيء. تمكن `points-per-batch` الاستدلال الموازي للنقاط في وضع تجزئة كل شيء. يسمح هذا باستدلال أسرع، ولكنه يستهلك ذاكرة أكبر. علاوة على ذلك، لا يدعم SAM المعالجة الدُفعية عبر الصور فقط ولكن عبر النقاط أيضًا. `pred_iou_thresh` هو عتبة ثقة IoU حيث يتم إرجاع الأقنعة الموجودة فوق عتبة معينة فقط.

تبدو الأقنعة كما يلي:

يمكننا تصورها على النحو التالي:

فيما يلي الصورة الأصلية باللون الرمادي مع خرائط ملونة. رائع!

## استدلال النموذج

### مطالبة النقطة

يمكنك أيضًا استخدام النموذج بدون خط الأنابيب. للقيام بذلك، قم بتهيئة النموذج والمعالج.

لتنفيذ مطالبة النقطة، قم بتمرير نقطة الإدخال إلى المعالج، ثم خذ إخراج المعالج ومرره إلى النموذج للاستدلال. لمعالجة إخراج النموذج، قم بتمرير الإخراج و`original_sizes` و`reshaped_input_sizes` التي نأخذها من الإخراج الأولي للمعالج. نحن بحاجة إلى تمرير هذه القيم لأن المعالج يقوم بإعادة تحجيم الصورة، ويجب استقراء الإخراج.

يمكننا تصور الأقنعة الثلاثة في إخراج "الأقنعة".

### مطالبة المربع

يمكنك أيضًا إجراء مطالبة المربع بطريقة مماثلة لمطالبة النقطة. يمكنك ببساطة تمرير مربع الإدخال بتنسيق قائمة `[x_min, y_min, x_max, y_max]` إلى جانب الصورة إلى المعالج. خذ إخراج المعالج ومرره مباشرةً إلى النموذج، ثم قم بمعالجة الإخراج مرة أخرى.

يمكنك تصور المربع المحيط بالنحلة كما هو موضح أدناه.

يمكنك رؤية إخراج الاستدلال أدناه.

يمكنك رؤية إخراج الاستدلال أدناه.