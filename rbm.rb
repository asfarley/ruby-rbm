class RBM
  @@L = 0.01
  
  def init_weights(input_vector_size, num_hidden_units)
    @weights = []
    for i in 0..(num_hidden_units-1)
      @weights[i] = Array.new(input_vector_size){ rand }
    end
  end
      
  def activation_energy(input, hidden_unit)
    sum = 0
    for i in 0..(@weights[hidden_unit].size-1)
      sum += @weights[hidden_unit][i]*input[i]
    end
    sum
  end
  
  def sigma(activation_energy)
   p = 1/(1+2.71**(-activation_energy))
  end
  
  def compute_activations(input)
    activations = []
      for i in 0..(@weights.size-1)
        p = sigma(activation_energy(input, i))
        activations[i] = rand < p ? 1 : 0
      end
    activations
  end
  
  def compute_agreement(input, activations)
    agreement = []
    for i in 0..(activations.size-1)
      a = []
      for j in 0..(input.size-1)
        a[j] = input[j]*activations[i]
      end
      agreement[i] = a
    end
    agreement
  end
  
  def reconstruct(activations)
    reconstruction = Array.new(@weights[0].size)
    for i in 0..(reconstruction.size-1)
    sum = 0
      for j in 0..(@weights.size-1)
        sum += @weights[j][i]*activations[j]
      end
    p = sigma(sum)
    reconstruction[i] = rand < p ? 1 : 0
    end
    reconstruction
  end
  
  def update_weights(input)
    activations = compute_activations(input)
    e_positive = compute_agreement(input, activations)
    reconstruction = reconstruct(activations)
    e_negative = compute_agreement(reconstruction, activations)
    
    for i in 0..(@weights.size-1)
      for j in 0..(@weights[i].size-1)
        @weights[i][j] = @weights[i][j] + @@L*(e_positive[i][j]-e_negative[i][j])
      end
    end
    
  end
  
  def weights
    @weights
  end
  
end

INPUT_VECTOR_SIZE = 5
NUM_HIDDEN_UNITS = 2

$rbm = RBM.new()
$rbm.init_weights(INPUT_VECTOR_SIZE, NUM_HIDDEN_UNITS)
training_samples = []

training_samples[0] = [1, 0, 0, 0, 0]
training_samples[1] = [0, 0, 0, 0, 1]

for i in 0..100000
  $rbm.update_weights(training_samples.sample)
end

p $rbm.weights