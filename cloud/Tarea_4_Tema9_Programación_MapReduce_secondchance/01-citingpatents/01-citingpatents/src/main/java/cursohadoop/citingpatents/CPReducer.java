package cursohadoop.citingpatents;

/**
 * Reducer para CitingPatents - cites by number: Obtiene el número de citas de una patente.
 * Para cada línea, obtiene la clave (patente) y une en un string el número de patentes que la citan,
 * ordenándolas primero numéricamente.
 */
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

// TODO: Completa la clase reducer
public class CPReducer extends Reducer<IntWritable, IntWritable, IntWritable, Text> {
	/**
	 * Método reduce
	 * @param key Patente citada
	 * @param values Lista con las patentes que la citan
	 * @param context Contexto MapReduce
	 * @throws IOException
	 * @throws InterruptedException
	 */
	// TODO: Completar el reducer
	@Override
	public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
		// Convierte los valores en una lista de Integers para ordenarlos numéricamente
		List<Integer> listaValores = new ArrayList<Integer>();
		for(IntWritable v : values) {
			listaValores.add(v.get());
		}
		// Ordena la lista de valores
		Collections.sort(listaValores);
		// Convierte la lista ordenada en un string separando por coma
		String csvciting = StringUtils.join(listaValores, ",");

		// Escribe la salida
		context.write(key, new Text(csvciting));
	}
}
