<main>
  <div class="canvas-container centered" style="width:{innerWidth*0.9}px; height:{innerHeight*0.9}px; background-color:cadetblue">
    
    <div class="centered" style="width:{canvasSize}px; height:{canvasSize}px">
        <P5 {sketch} />
    </div>
  </div>
</main>

<svelte:window bind:innerWidth bind:innerHeight />

<script>
  import p5Svelte from 'p5-svelte';
  import P5 from 'p5-svelte';

  let innerWidth = 0;
  let innerHeight = 0;
  
  $: condition = innerWidth*1.33 <= innerHeight

  let circleWidth = 55;
  let circleHeight = 55;

  let canvasWidth = 0;
  let canvasHeight = 0;

  let bufferWidth = 1000;
  let bufferHeight = 1000;

  $: canvasWidth = innerWidth*0.9;
  $: canvasHeight = innerHeight*0.9;

  $: canvasSize = Math.min(canvasWidth, canvasHeight);

  var drawingBuffer;

  const sketch = (p5) => {
    p5.draw = () => {
      if (p5.mouseIsPressed) {
        drawingBuffer.fill(0);
        drawingBuffer.ellipse(p5.mouseX/canvasSize*bufferWidth, p5.mouseY/canvasSize*bufferHeight, 80, 80);
      } 

      p5.image(drawingBuffer, 0, 0, canvasSize, canvasSize);
    };


    p5.setup = () => {
      p5.createCanvas(canvasSize, canvasSize);
      p5.background(200, 0, 100);
      drawingBuffer = p5.createGraphics(bufferWidth, bufferHeight);
      drawingBuffer.background(200, 0, 100);
    };

    

    p5.windowResized= () => {
      p5.resizeCanvas(canvasSize, canvasSize, true);
    }
  };


</script>



<style>
  .canvas-container{
    position: absolute;
  }

  .centered{     
    position: absolute;
    top:0;
    bottom: 0;
    left: 0;
    right: 0;
    
    margin: auto;
  }
</style>



<!-- <script>
  import svelteLogo from './assets/svelte.svg'
  import Counter from './lib/Counter.svelte'
</script>

<main>
  <div>
    <a href="https://vitejs.dev" target="_blank" rel="noreferrer"> 
      <img src="/vite.svg" class="logo" alt="Vite Logo" />
    </a>
    <a href="https://svelte.dev" target="_blank" rel="noreferrer"> 
      <img src={svelteLogo} class="logo svelte" alt="Svelte Logo" />
    </a>
  </div>
  <h1>Vite + Svelte</h1>

  <div class="card">
    <Counter />
  </div>

  <p>
    Check out <a href="https://github.com/sveltejs/kit#readme" target="_blank" rel="noreferrer">SvelteKit</a>, the official Svelte app framework powered by Vite!
  </p>

  <p class="read-the-docs">
    Click on the Vite and Svelte logos to learn more
  </p>
</main>

<style>
  .logo {
    height: 6em;
    padding: 1.5em;
    will-change: filter;
    transition: filter 300ms;
  }
  .logo:hover {
    filter: drop-shadow(0 0 2em #646cffaa);
  }
  .logo.svelte:hover {
    filter: drop-shadow(0 0 2em #ff3e00aa);
  }
  .read-the-docs {
    color: #888;
  }
</style>
 -->
