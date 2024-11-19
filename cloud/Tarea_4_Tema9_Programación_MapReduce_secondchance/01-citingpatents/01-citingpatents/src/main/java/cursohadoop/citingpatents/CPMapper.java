package cursohadoop.citingpatents;

/**
 * Mapper para CitingPatents - cites by number: Obtiene el número de citas de una patente.
 * Para cada línea, invierte las columnas (patente citante, patente a la que cita)
 * conviertiéndolas primero en enteros para tener una ordenación numérica por clave
 *
 * Entrada: 
 *   Clave: Patente citante (String)
 *   Valor: Patente citada (String)
 * Salida:
 *   Clave: Patente citada (entero)
 *   Valor: Patente citante (entero)
 *
 */
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import java.io.IOException;

public class CPMapper extends Mapper<Text, Text, IntWritable, IntWritable> {
	/*
	 * Método map
	 * @param key patente que cita
	 * @param value patente citada
	 * @param context Contexto MapReduce
	 * @throws IOException
	 * 
	 * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN,
	 * org.apache.hadoop.mapreduce.Mapper.Context)
	 */
	@Override
	public void map(Text key, Text value, Context context) throws IOException,
			InterruptedException {

		// to skip the header
		if (key.toString().equals("CITING") || value.toString().equals("CITED")) {
			return;
		}
		
		try {
			//Get key and value as integers
			int citing = Integer.parseInt(key.toString().trim());
			int cited = Integer.parseInt(value.toString().trim());

			//Set the key and value into the IntWritable objects
			citingWritable.set(citing);
			citedWritable.set(cited);

			//Emit the inverted relationship
			context.write(citedWritable, citingWritable);
		} catch (NumberFormatException e) {
			// Handle the exception if the key or value cannot be parsed as an integer
			System.err.println("Invalid number format: " + e.getMessage());
		}
	}
	private IntWritable citedWritable = new IntWritable();
    private IntWritable citingWritable = new IntWritable();
}
