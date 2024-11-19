package cursohadoop.citationnumberbypatent_chained;

/**
 *  CNBP_a - cites number by patent: Obtiene el número de citas de una patente 
 *     Combina mapper | reducer | mapper
 *     
 *  	mapper1 -> CPMapper
 *                Para cada línea, invierte las columnas (patente citada, patente que cita)
 *      reducer -> CPReducer
 *     		  Para cada línea, obtiene la clave (patente) y une en un string las patentes que la citan
 *      mapper2 -> CCMapper
 *     		  De la salida del reducer, para cada patente cuenta el número de patentes que la citan
 *      
 */

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.mapreduce.lib.chain.ChainMapper;
import org.apache.hadoop.mapreduce.lib.chain.ChainReducer;

import cursohadoop.citingpatents.CPMapper;
import cursohadoop.citingpatents.CPReducer;

public class CNBPDriver_chained extends Configured implements Tool {
	/*
	 * (non-Javadoc)
	 * 
	 * @see org.apache.hadoop.util.Tool#run(java.lang.String[])
	 */
	@Override
	public int run(String[] arg0) throws Exception {

		/*
		 * Comprueba los parámetros de entrada  
		 */
		if (arg0.length != 2) {
			System.err.printf("Usar: %s [opciones genéricas] <directorio_entrada> <directorio_salida>%n",
					getClass().getSimpleName());
			System.err.printf("Recuerda que el directorio de salida no puede existir");
			ToolRunner.printGenericCommandUsage(System.err);
			return -1;
		}
		
		//Obtiene la configuración por defecto
		Configuration conf = getConf();	
		//TODO: Modifica el separador clave valor de entrada	
		conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator", ",");
				
		/* Define el job */
		Job job = Job.getInstance(conf);
		job.setJobName("Trabajo encadenado");

		/* Fija el jar del trabajo a partir de la clase del objeto actual */
		job.setJarByClass(getClass());

		// TODO: Añade al job los paths de entrada y salida */
		FileInputFormat.addInputPath(job, new Path(arg0[0]));
		FileOutputFormat.setOutputPath(job, new Path(arg0[1]));
		
	
		// TODO Fija el formato de los ficheros de entrada y salida 
		job.setInputFormatClass(KeyValueTextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		
		// TODO Especifica el primer mapper
		/* El booleano (true) especifica si los datos en la cadena se pasan por valor (true) o referencia (false) */
		//ChainMapper.addMapper(CPMapper.class, new Configuration(false));
		// Add first Mapper to chain
		Configuration map1Conf = new Configuration(false);
		ChainMapper.addMapper(
			job,
			CPMapper.class,
			Text.class, Text.class,
			IntWritable.class, IntWritable.class,
			map1Conf
		);
		
		// TODO Añade el reducer */
		//ChainReducer.setReducer(CPReducer.class, new Configuration(false));
        // Add Reducer to chain
        Configuration reduceConf = new Configuration(false);
        ChainReducer.setReducer(
            job,
            CPReducer.class,
            IntWritable.class, IntWritable.class,
            IntWritable.class, Text.class,
            reduceConf
        );

		
		// TODO: El siguiente mapper se concatenan al reducer
		//ChainReducer.addMapper(CCMapper.class, new Configuration(false));
		// Add second Mapper to chain
		Configuration map2Conf = new Configuration(false);
		ChainReducer.addMapper(
			job,
			CCMapper.class,
			IntWritable.class, Text.class,
			IntWritable.class, IntWritable.class,
			map2Conf
		);

		// Especifica el número de reducers
		job.setNumReduceTasks(2);
		
		// Lanza el trabajo a ejecución
		return job.waitForCompletion(true) ? 0 : 1;
	}

	/**
	 * Usar yarn jar CitationNumberByPatent_chained.jar dir_entrada dir_salida
	 * 
	 * @param args
	 *            dir_entrada dir_salida
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		int exitCode = ToolRunner.run(new Configuration(), new CNBPDriver_chained(), args);
		System.exit(exitCode);
	}

}

