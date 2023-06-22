require 'dotenv'
Dotenv.load
require 'daru'
require "numo/narray"
require './openai_utils'
require 'benchmark'

DOC_EMBEDDINGS_MODEL_E = "gpt-4"
QUERY_EMBEDDINGS_MODEL_E = "gpt-4"

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

openai = OpenAI_API.new(DOC_EMBEDDINGS_MODEL_E, QUERY_EMBEDDINGS_MODEL_E)

original_email = "Hola, Soy Ana. Podrías explicarme cómo debo darme de baja el impuesto de circulacón. Saludos, Ana."

prompt = "Responde a este 'Email:' tan correctamente como sea posible usando el 'Context:' como fuente de información, y si no puedes generarla, di \"I don't know\"

Context: 
#{context}


Email:
#{original_email}

Hola, 
"

answer = openai.generate_answer(prompt)
p answer

# " Para solicitar la baja del impuesto de circulación, tendrás que dirigirte a las oficinas del Ayuntamiento (específicamente a la ventanilla de recaudación) y rellenar un impreso de solicitud de baja. Ten en cuenta que debes llevar contigo el justificante de baja definitiva de la DGT y el Certificado de Destrucción."


context = "A través de esta opción podrás realizar tu solicitud de prestaciones por desempleo en línea y sin necesidad de desplazamientos.

SOLO PARA ACCESOS A TRAVÉS DE CL@VE: Para poder firmar un trámite en el SEPE, debe haberse realizado el registro de nivel avanzado en el sistema Cl@ve, bien de forma presencial, en una oficina ante un empleado público habilitado al efecto, o bien de forma telemática, previa autenticación del ciudadano mediante un certificado electrónico reconocido
ATENCION: La nueva versión de solicitud a través de la sede electrónica simplifica y actualiza los tratamientos, unificando y reduciendo pantallas e incorporando mejoras en los avisos de finalización de los trámites.
Formulario para presolicitud individual de prestaciones por desempleo.
Solicitud de prestación contributiva. También puedes reconocer tu prestación en el momento a través de la web.
Solicitud de subsidio por desempleo. También puedes realizar la prórroga de tu subsidio en el momento a través de la web.
Solicitud de renta activa de inserción. Puede obtener digitalmente el Certificado de Emigrante Retornado en este enlace.
Solicitud del abono acumulado y anticipado para personas extranjeras no comunitarias.
Solicitud de pago único de la prestación por desempleo.
Solicitud ayuda suplementaria RAI para víctimas de violencia de género, sexual o doméstica.
Solicitud de Renta Agraria para trabajadores eventuales agrarios en Andalucía y Extremadura.
Solicitud de Subsidio agrícola para Trabajadores Eventuales del SEASS.
Para la solicitud de prestación por desempleo, el trámite podrá completarse con el anexado de la documentación que se indica en el resguardo de solicitud que el sistema facilita al finalizar la gestión.

Tu solicitud quedará registrada telemáticamente, lo que otorga validez a tu presentación como si ésta se hubiera efectuado de forma presencial. Para que este registro pueda efectuarse desde tu ordenador, es precisa la instalación de un componente adicional (ActiveX) y la configuración de tu navegador según se describe en las siguientes instrucciones: Configuración navegador (pdf - 632 KB).

Para poder tramitar solicitudes de prestaciones por desempleo a través de la Sede Electrónica del SEPE, es imprescindible acceder a través de certificado digital o DNIe o usuario y contraseña obtenida a través del sistema de cl@ve y estar inscrito como demandante de empleo autonómico o, si reside en Ceuta y Melilla, en el Servicio Público de Empleo Estatal (SEPE).

Para finalizar los trámites que requieren firma, es imprescindible, cuando se accede con usuario y contraseña cl@ve, disponer de teléfono móvil y que coincida con el registrado en cl@ve.

Antes de realizar tu solicitud, puedes consultar cuál es la que corresponde a tu situación de desempleo utilizando este programa de simulación de prestaciones.

Importante: Si eres trabajador del mar puedes utilizar este servicio de solicitud por Internet para tramitar tu prestación. Para más información consulta la red de Oficinas de Empleo Marítimo.

"

openai = OpenAI_API.new(DOC_EMBEDDINGS_MODEL_E, QUERY_EMBEDDINGS_MODEL_E)

original_email = "Hola, Soy Daniel. Siendo autónomo ¿cómo debo solicitar la prestación por desempleo y dónde? Saludos, Daniel."

prompt = "Responde a este 'Email:' tan correctamente como sea posible usando el 'Context:' como fuente de información, y si no puedes generarla, di \"I don't know\"

Context: 
#{context}


Email:
#{original_email}

Hola, 
"
Benchmark.bm do |x|
  x.report("answer") {
    answer = openai.generate_answer(prompt)
  }
  p answer
end

# "text-davinci-003" 4.62
# "text-curie-001" 6.1
# "text-ada-001" 4.20
# "ada" 4.89
# "gpt-3.5-turbo" 3.9
# "text-davinci-002" 4.07
# "gpt-4" 4.14