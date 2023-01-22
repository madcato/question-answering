require 'dotenv'
Dotenv.load
require 'daru'
require "numo/narray"
require './openai_utils'

context = "Cuando hemos decidido dar de baja nuestro vehículo, también tendremos que dar de baja el impuesto de circulación o IVTM, si es que lo hacemos después de que finalice el año en curso.

Para solicitar la baja del impuesto de circulación, tendremos que dirigirnos a las oficinas del Ayuntamiento (específicamente a la ventanilla de recaudación) y rellenar un impreso de solicitud de baja. Tenemos de llevar con nosotros el justificante de baja definitiva de la DGT y el Certificado de Destrucción.

Cómo solicitar la devolución del IVTM al Ayuntamiento
En el caso de que hayamos dado de baja definitiva nuestro vehículo tras haber pagado el impuesto de circulación, podremos solicitar la devolución de una parte de la cuota pagada tanto si hemos tramitado una baja definitiva como una temporal.

La cuota del IVTM se prorratea por trimestres naturales en función de los que restan por vencer en caso de nuevas adquisiciones, bajas definitivas y bajas temporales en la Dirección General de Tráfico.

Esto significa que, si por ejemplo, hemos pagado al Ayuntamiento un año completo en febrero y das de baja tu vehículo en abril, podrás pedir la devolución del impuesto que no se ha disfrutado durante el tercer y cuarto trimestre.

Para solicitar la devolución del impuesto deberás:

Rellenar el impreso de solicitud de devolución del IVTM, disponible en la sección de recaudación del Ayuntamiento en el que esté registrado tu vehículo
Adjuntar a la solicitud el justificante de baja, ya sea temporal o definitiva en la DGT
Aportar  el Certificado de Destrucción del vehículo, en el caso de que éste haya sido enviado al desguace"

openai = OpenAI_API.new(DOC_EMBEDDINGS_MODEL, QUERY_EMBEDDINGS_MODEL)

original_email = "Hola, Soy Ana. Podrías explicarme cómo debo darme de baja el impuesto de circulacón. Saludos, Ana."

prompt = "Responde a este correo tan correctamente como sea posible, y si no puedes generarla con el siguiente texto 'context:', say \"I don't know\"

Context:
#{context}

Q: #{original_email}
A:"

answer = openai.generate_answer(prompt)

p answer

# " Para solicitar la baja del impuesto de circulación, tendrás que dirigirte a las oficinas del Ayuntamiento (específicamente a la ventanilla de recaudación) y rellenar un impreso de solicitud de baja. Ten en cuenta que debes llevar contigo el justificante de baja definitiva de la DGT y el Certificado de Destrucción."
