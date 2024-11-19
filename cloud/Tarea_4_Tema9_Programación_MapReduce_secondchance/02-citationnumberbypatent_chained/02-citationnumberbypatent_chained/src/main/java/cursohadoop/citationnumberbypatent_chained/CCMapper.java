package cursohadoop.citationnumberbypatent_chained;

/**
 * Mapper Count Cites 
 * Para cada línea, obtiene la clave (patente) y cuenta el número de patentes que la citan
 */
import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

// TODO: Completa la clase Mapper
public class CCMapper extends Mapper<IntWritable, Text, IntWritable, IntWritable> {
	private IntWritable count = new IntWritable();


	@Override
	public void map(IntWritable key, Text value, Context context)
			throws IOException, InterruptedException {
	    		
		StringTokenizer tokenizer = new StringTokenizer(value.toString(), ",");
        int citingCount = tokenizer.countTokens();
        count.set(citingCount);
        context.write(key, count);
	}

}
