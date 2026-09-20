from flask import Flask, render_template, request, send_file, Response
from flask_socketio import SocketIO, emit
import json
import subprocess
import signal
import sys
import os
from devotion_generator import DevotionGenerator
from tts_service import tts_service

app = Flask(__name__)
app.config['SECRET_KEY'] = 'devotion-secret-key'

# Configure SocketIO for Cloudflare proxy compatibility
try:
    import gevent
    socketio = SocketIO(
        app, 
        cors_allowed_origins="*", 
        async_mode='gevent',
        ping_timeout=60,
        ping_interval=25,
        engineio_logger=False,
        socketio_logger=False
    )
except ImportError:
    socketio = SocketIO(
        app, 
        cors_allowed_origins="*",
        ping_timeout=60,
        ping_interval=25,
        engineio_logger=False,
        socketio_logger=False
    )

generator = DevotionGenerator()

llama_process = None


def start_llama_server():
    global llama_process
    print("Starting llama server...")
    llama_process = subprocess.Popen(
        ['llama', 'serve', '-m', '/media/novadev/storage/daylivotion/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf', '--port', '8981', '--host', '127.0.0.1'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to be ready
    print("Waiting for llama server to start...")
    while True:
        line = llama_process.stderr.readline()
        if not line:
            print("Llama server process ended unexpectedly")
            return False
        if 'listening on http://127.0.0.1:8981' in line:
            print("Llama server is ready!")
            return True
        if llama_process.poll() is not None:
            print("Llama server process died")
            return False


def cleanup_and_exit(signum=None, frame=None):
    global llama_process
    print("Cleaning up before exit...")
    if llama_process:
        print("Stopping llama server...")
        llama_process.terminate()
        try:
            llama_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Force killing llama server...")
            llama_process.kill()
            llama_process.wait()
    print("Exit complete")
    sys.exit(0)


# Setup signal handlers
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sitemap.xml')
def sitemap():
    return send_file('static/sitemap.xml', mimetype='application/xml')

@app.route('/generate_audio', methods=['POST', 'OPTIONS'])
def generate_audio():
    """Generate audio for a devotion."""
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response
    
    try:
        data = request.json
        reference = data.get('reference')
        content = data.get('content')
        devotion = data.get('devotion')
        prayer = data.get('prayer')
        sid = data.get('sid')  # Get socket ID from client
        
        if not all([reference, content, devotion, prayer]):
            return {'error': 'Missing required fields'}, 400
        
        if not tts_service.is_available():
            return {'error': 'TTS service not available'}, 503
        
        audio_path = tts_service.generate_audio(reference, content, devotion, prayer, socketio, sid)
        
        # Send file with proper CORS headers for Safari/iOS compatibility
        response = send_file(audio_path, mimetype='audio/wav', as_attachment=False)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'no-cache'
        
        @response.call_on_close
        def cleanup():
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                print(f"Error cleaning up audio file: {e}")
        
        return response
    except Exception as e:
        error_response = {'error': str(e)}
        error_response_obj = Response(json.dumps(error_response), status=500, mimetype='application/json')
        error_response_obj.headers['Access-Control-Allow-Origin'] = '*'
        return error_response_obj

@socketio.on('generate_devotion')
def handle_generate_devotion():
    try:
        gen = generator.generate_devotion(request.sid)
        for progress in gen:
            emit('progress', progress)
    except Exception as e:
        emit('progress', {'status': 'error', 'error': f'Server error: {str(e)}'})

@socketio.on('disconnect')
def handle_disconnect():
    # Remove client from devotion queue if they're waiting
    generator.remove_client_from_queue(request.sid)
    # Cancel devotion generation if they're currently generating
    generator.cancel_generation(request.sid)
    # Remove client from TTS queue if they're waiting
    tts_service.remove_client_from_tts_queue(request.sid)
    # Cancel TTS generation if they're currently generating
    tts_service.cancel_tts_generation(request.sid)


if __name__ == '__main__':
    # Only start llama server in the main process (not the reloader)
    # Flask debug mode starts the app twice - this prevents duplicate llama servers
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        # Start llama server and wait for it to be ready
        if not start_llama_server():
            print("Failed to start llama server")
            sys.exit(1)
    
    try:
        # Run the main Flask server
        socketio.run(app, host='0.0.0.0', port=5100, debug=True)
    finally:
        cleanup_and_exit()
